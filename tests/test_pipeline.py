from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.lib.cli_parse import parse_pipeline_expression
from src.lib.packets import FrameMetadata, FramePacket, new_packet_id
from src.lib.pipeline import ConnectionSpec, ElementSpec, Pipeline, PipelineSpec
from src.lib.registry import register_builtin_elements
from src.sinks.displaysink import DisplaySink
from src.transformers.combine import Combine
from src.transformers.hist_equalize import HistEqualize
from src.transformers.resize import Resize


def packet(
    data: np.ndarray,
    *,
    stream_id: str = "s",
    source_id: str = "src",
    index: int = 0,
    fmt: str = "bgr",
) -> FramePacket:
    height, width = data.shape[:2]
    channels = 1 if data.ndim == 2 else data.shape[2]
    return FramePacket(
        data=data,
        metadata=FrameMetadata(
            packet_id=new_packet_id(),
            stream_id=stream_id,
            source_id=source_id,
            pts=float(index) / 30.0,
            index=index,
            format=fmt,
            width=width,
            height=height,
            fps=30.0,
            depth=data.dtype.itemsize * 8,
            channels=channels,
        ),
    )


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        register_builtin_elements()

    def test_frame_packet_rejects_raw_metadata(self) -> None:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with self.assertRaises(TypeError):
            FramePacket(data=frame, metadata={"width": 4})  # type: ignore[arg-type]

    def test_resize_updates_metadata_and_parents(self) -> None:
        transform = Resize("r", {"width": 8, "height": 6})
        source = packet(np.zeros((4, 5, 3), dtype=np.uint8))
        result = transform.process({"in": source})["out"][0]
        self.assertEqual(result.metadata.width, 8)
        self.assertEqual(result.metadata.height, 6)
        self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_hist_equalize_supports_bgr_8bit(self) -> None:
        transform = HistEqualize("eq", {"mode": "global"})
        frame = np.full((8, 8, 3), 64, dtype=np.uint8)
        result = transform.process({"in": packet(frame)})["out"][0]
        self.assertEqual(result.metadata.depth, 8)
        self.assertEqual(result.metadata.format, "bgr")

    def test_combine_horizontal_preserves_both_parents(self) -> None:
        left = packet(np.zeros((4, 5, 3), dtype=np.uint8), stream_id="l")
        right = packet(np.ones((4, 6, 3), dtype=np.uint8), stream_id="r")
        transform = Combine("c", {"mode": "horizontal"})
        result = transform.process({"left": left, "right": right})["out"][0]
        self.assertEqual(result.metadata.width, 11)
        self.assertEqual(result.metadata.height, 4)
        self.assertIn(left.metadata.packet_id, result.metadata.parents)
        self.assertIn(right.metadata.packet_id, result.metadata.parents)

    def test_combine_rejects_depth_mismatch(self) -> None:
        left = packet(np.zeros((4, 5, 3), dtype=np.uint8))
        right = packet(np.zeros((4, 5, 3), dtype=np.uint16))
        transform = Combine("c", {"mode": "horizontal"})
        with self.assertRaises(ValueError):
            transform.process({"left": left, "right": right})

    def test_cli_parser_linear(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mp4 ! resize width=4 height=4 ! filesink path=out.mp4"
        )
        self.assertEqual([element.type for element in spec.elements], [
            "filesrc",
            "resize",
            "filesink",
        ])
        self.assertEqual(len(spec.connections), 2)

    def test_cli_parser_named_graph(self) -> None:
        spec = parse_pipeline_expression(
            """
            filesrc name=a path=a.mp4 ! resize name=ra width=4 height=4
            filesrc name=b path=b.mp4 ! resize name=rb width=4 height=4
            ra.out ! combine.left name=c mode=horizontal
            rb.out ! c.right
            c.out ! filesink path=out.mp4
            """
        )
        ids = {element.id for element in spec.elements}
        self.assertTrue({"a", "b", "ra", "rb", "c", "filesink"}.issubset(ids))
        self.assertIn(ConnectionSpec("rb", "out", "c", "right"), spec.connections)

    def test_displaysink_uses_explicit_or_metadata_fps(self) -> None:
        frame = packet(np.zeros((4, 5, 3), dtype=np.uint8))
        metadata_fps_sink = DisplaySink("display", {"enabled": False})
        explicit_fps_sink = DisplaySink("display", {"enabled": False, "fps": 25})
        wait_sink = DisplaySink("display", {"enabled": False, "wait_ms": 7})

        self.assertEqual(metadata_fps_sink._wait_ms(frame), 33)
        self.assertEqual(explicit_fps_sink._wait_ms(frame), 40)
        self.assertEqual(wait_sink._wait_ms(frame), 7)

    def test_pipeline_writes_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.mp4"
            output_path = tmp_path / "output.mp4"
            _write_test_video(input_path, frame_count=3, size=(16, 12))

            spec = PipelineSpec(
                elements=[
                    ElementSpec("src", "filesrc", {"path": str(input_path)}),
                    ElementSpec("resize", "resize", {"width": 8, "height": 6}),
                    ElementSpec("eq", "hist_equalize", {"mode": "global"}),
                    ElementSpec("out", "filesink", {"path": str(output_path)}),
                ],
                connections=[
                    ConnectionSpec("src", "out", "resize", "in"),
                    ConnectionSpec("resize", "out", "eq", "in"),
                    ConnectionSpec("eq", "out", "out", "in"),
                ],
            )
            Pipeline.from_spec(spec).run()
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


def _write_test_video(path: Path, frame_count: int, size: tuple[int, int]) -> None:
    width, height = size
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create test video")
    for index in range(frame_count):
        frame = np.full((height, width, 3), index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    unittest.main()
