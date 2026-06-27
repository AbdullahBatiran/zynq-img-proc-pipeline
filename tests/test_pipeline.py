from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.cli import main as cli_main
from src.lib.cli_parse import parse_pipeline_expression
from src.lib.elements import PipelineContext
from src.lib.packets import FrameMetadata, FramePacket, new_packet_id
from src.lib.pipeline import ConnectionSpec, ElementSpec, Pipeline, PipelineSpec
from src.lib.registry import register_builtin_elements
from src.sinks.displaysink import DisplaySink
from src.sources.filesrc import infer_frame_format, normalize_decoded_frame
from src.transformers.combine import Combine
from src.transformers.hist_equalize import HistEqualize
from src.transformers.linear_scale import LinearScale
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

    def test_filesrc_infers_mono_uint16_metadata_shape(self) -> None:
        frame = np.zeros((4, 5, 1), dtype=np.uint16)
        normalized, frame_format = normalize_decoded_frame(
            frame=frame,
            requested_format="gray",
            requested_depth=16,
            strict=True,
            path="mono16.mkv",
        )
        self.assertEqual(normalized.shape, (4, 5))
        self.assertEqual(normalized.dtype, np.uint16)
        self.assertEqual(frame_format, "gray")
        self.assertEqual(infer_frame_format(normalized), "gray")

    def test_filesrc_infers_three_channel_uint8(self) -> None:
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        normalized, frame_format = normalize_decoded_frame(
            frame=frame,
            requested_format="bgr",
            requested_depth="auto",
            strict=True,
            path="color.mp4",
        )
        self.assertEqual(normalized.shape, (4, 5, 3))
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(frame_format, "bgr")

    def test_filesrc_strict_mono16_rejects_bgr_8bit(self) -> None:
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            normalize_decoded_frame(
                frame=frame,
                requested_format="gray",
                requested_depth=16,
                strict=True,
                path="mono16.mkv",
            )

    def test_hist_equalize_supports_bgr_8bit(self) -> None:
        transform = HistEqualize("eq", {"bins": 256})
        frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape((8, 8, 3))
        source = packet(frame)
        result = transform.process({"in": source})["out"][0]
        self.assertEqual(result.metadata.depth, 8)
        self.assertEqual(result.metadata.format, "bgr")
        self.assertEqual(result.metadata.extra["hist_bins"], 256)
        self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_hist_equalize_supports_gray_8bit(self) -> None:
        transform = HistEqualize("eq", {"bins": 128})
        frame = np.arange(64, dtype=np.uint8).reshape((8, 8))
        result = transform.process({"in": packet(frame, fmt="gray")})["out"][0]
        self.assertEqual(result.data.dtype, np.uint8)
        self.assertEqual(result.metadata.channels, 1)
        self.assertEqual(result.metadata.extra["hist_bins"], 128)

    def test_hist_equalize_supports_gray_16bit(self) -> None:
        transform = HistEqualize("eq", {"bins": 1024})
        frame = np.linspace(0, 65535, 64, dtype=np.uint16).reshape((8, 8))
        result = transform.process({"in": packet(frame, fmt="gray")})["out"][0]
        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.metadata.depth, 16)
        self.assertEqual(result.metadata.extra["hist_bins"], 1024)

    def test_hist_equalize_supports_rgb_16bit(self) -> None:
        transform = HistEqualize("eq", {})
        frame = np.linspace(0, 65535, 8 * 8 * 3, dtype=np.uint16).reshape((8, 8, 3))
        result = transform.process({"in": packet(frame, fmt="rgb")})["out"][0]
        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.metadata.format, "rgb")
        self.assertEqual(result.metadata.depth, 16)
        self.assertEqual(result.metadata.extra["hist_bins"], 65536)

    def test_hist_equalize_rejects_unsupported_depth(self) -> None:
        transform = HistEqualize("eq", {})
        frame = np.zeros((8, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            transform.process({"in": packet(frame, fmt="gray")})

    def test_hist_equalize_rejects_unsupported_channels(self) -> None:
        transform = HistEqualize("eq", {})
        frame = np.zeros((8, 8, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            transform.process({"in": packet(frame, fmt="bgr")})

    def test_linear_scale_default_uint8_maps_min_max(self) -> None:
        transform = LinearScale("scale", {})
        frame = np.array([[10, 15], [20, 30]], dtype=np.uint8)
        source = packet(frame, fmt="gray")
        result = transform.process({"in": source})["out"][0]

        self.assertEqual(result.data.dtype, np.uint8)
        self.assertEqual(int(result.data.min()), 0)
        self.assertEqual(int(result.data.max()), 255)
        self.assertEqual(result.metadata.depth, 8)
        self.assertEqual(result.metadata.extra["linear_scale_input_min"], 10.0)
        self.assertEqual(result.metadata.extra["linear_scale_input_max"], 30.0)
        self.assertEqual(result.metadata.extra["linear_scale_output_type"], "uint8")
        self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_linear_scale_default_uint16_maps_min_max(self) -> None:
        transform = LinearScale("scale", {})
        frame = np.array([[100, 1100], [2100, 4100]], dtype=np.uint16)
        result = transform.process({"in": packet(frame, fmt="gray")})["out"][0]

        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(int(result.data.min()), 0)
        self.assertEqual(int(result.data.max()), 65535)
        self.assertEqual(result.metadata.depth, 16)

    def test_linear_scale_otype_uint8_updates_metadata_depth(self) -> None:
        transform = LinearScale("scale", {"otype": "uint8"})
        frame = np.array([[0, 32768], [49152, 65535]], dtype=np.uint16)
        result = transform.process({"in": packet(frame, fmt="gray")})["out"][0]

        self.assertEqual(result.data.dtype, np.uint8)
        self.assertEqual(result.metadata.depth, 8)
        self.assertEqual(result.metadata.extra["linear_scale_output_max"], 255.0)
        self.assertEqual(result.metadata.extra["linear_scale_output_type"], "uint8")

    def test_linear_scale_custom_output_range(self) -> None:
        transform = LinearScale("scale", {"omin": 10, "omax": 20})
        frame = np.array([[0, 50], [100, 200]], dtype=np.uint8)
        result = transform.process({"in": packet(frame, fmt="gray")})["out"][0]

        self.assertEqual(result.data.tolist(), [[10, 12], [15, 20]])
        self.assertEqual(result.metadata.extra["linear_scale_output_min"], 10.0)
        self.assertEqual(result.metadata.extra["linear_scale_output_max"], 20.0)

    def test_linear_scale_supplied_min_max_and_one_sided_ranges(self) -> None:
        frame = np.array([[10, 20], [30, 40]], dtype=np.uint8)

        with_min = LinearScale("scale", {"min": 0}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]
        with_max = LinearScale("scale", {"max": 50}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]
        with_both = LinearScale("scale", {"min": 10, "max": 40}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]

        self.assertEqual(with_min.metadata.extra["linear_scale_input_min"], 0.0)
        self.assertEqual(with_min.metadata.extra["linear_scale_input_max"], 40.0)
        self.assertEqual(with_max.metadata.extra["linear_scale_input_min"], 10.0)
        self.assertEqual(with_max.metadata.extra["linear_scale_input_max"], 50.0)
        self.assertEqual(with_both.data.tolist(), [[0, 85], [170, 255]])

    def test_linear_scale_percentile_ranges(self) -> None:
        frame = np.array([0, 10, 20, 30, 40], dtype=np.uint8).reshape((1, 5))

        symmetric = LinearScale("scale", {"perc": 0.25}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]
        lower_only = LinearScale("scale", {"perc-down": 0.25}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]
        upper_only = LinearScale("scale", {"perc_up": 0.25}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]

        self.assertEqual(symmetric.metadata.extra["linear_scale_input_min"], 10.0)
        self.assertEqual(symmetric.metadata.extra["linear_scale_input_max"], 30.0)
        self.assertEqual(lower_only.metadata.extra["linear_scale_input_min"], 10.0)
        self.assertEqual(lower_only.metadata.extra["linear_scale_input_max"], 40.0)
        self.assertEqual(upper_only.metadata.extra["linear_scale_input_min"], 0.0)
        self.assertEqual(upper_only.metadata.extra["linear_scale_input_max"], 30.0)

    def test_linear_scale_invalid_parameter_combinations(self) -> None:
        invalid_params = [
            {"min": 1, "perc": 0.1},
            {"min": 1, "perc-down": 0.1},
            {"max": 10, "perc": 0.1},
            {"max": 10, "perc-up": 0.1},
            {"perc": 0.1, "perc-down": 0.1},
            {"perc": 0.1, "perc-up": 0.1},
            {"perc": -0.1},
            {"perc-up": 1.1},
            {"perc_up": 0.1, "perc-up": 0.2},
        ]
        for params in invalid_params:
            with self.subTest(params=params), self.assertRaises(ValueError):
                LinearScale("scale", params)

        frame = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            LinearScale("scale", {"omin": 5, "omax": 5}).process(
                {"in": packet(frame, fmt="gray")}
            )
        with self.assertRaises(ValueError):
            LinearScale("scale", {"min": 10, "max": 1}).process(
                {"in": packet(frame, fmt="gray")}
            )

    def test_linear_scale_three_channel_uses_global_range(self) -> None:
        transform = LinearScale("scale", {})
        frame = np.array(
            [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 210]]],
            dtype=np.uint8,
        )
        result = transform.process({"in": packet(frame, fmt="rgb")})["out"][0]

        self.assertEqual(result.data.shape, frame.shape)
        self.assertEqual(result.metadata.channels, 3)
        self.assertEqual(result.metadata.format, "rgb")
        self.assertEqual(int(result.data[0, 0, 0]), 0)
        self.assertEqual(int(result.data[0, 0, 1]), 13)
        self.assertEqual(int(result.data.max()), 255)

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

    def test_cli_parser_accepts_hyphenated_linear_scale_params(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mp4 ! linear-scale otype=uint8 perc-up=0.01 "
            "! filesink path=out.mp4"
        )
        self.assertEqual(spec.elements[1].type, "linear-scale")
        self.assertEqual(spec.elements[1].params["perc-up"], 0.01)

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

    def test_cli_list_elements_verbose_shows_parameter_names(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["list-elements", "--verbose"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("filesrc:", output)
        self.assertIn("params=[path, stream_id, source_id", output)
        self.assertIn("hist_equalize:", output)
        self.assertIn("params=[bins]", output)
        self.assertIn("linear-scale:", output)
        self.assertIn(
            "params=[otype, omin, omax, min, max, perc, perc-down, perc-up]",
            output,
        )

    def test_cli_describe_linear_scale_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "linear-scale"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: linear-scale", output)
        self.assertIn("otype: str | optional", output)
        self.assertIn("perc-up: float | optional", output)
        self.assertIn("formats=[bgr, gray, rgb]", output)

    def test_cli_describe_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "filesrc"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: filesrc", output)
        self.assertIn("Parameters:", output)
        self.assertIn("path: path | required", output)
        self.assertIn("format: str | optional", output)
        self.assertIn("Output ports:", output)

    def test_cli_describe_unknown_element_returns_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = cli_main(["describe", "missing"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Unknown element 'missing'", stderr.getvalue())

    def test_displaysink_uses_explicit_or_metadata_fps(self) -> None:
        frame = packet(np.zeros((4, 5, 3), dtype=np.uint8))
        metadata_fps_sink = DisplaySink("display", {"enabled": False})
        explicit_fps_sink = DisplaySink("display", {"enabled": False, "fps": 25})
        wait_sink = DisplaySink("display", {"enabled": False, "wait_ms": 7})

        self.assertEqual(metadata_fps_sink._wait_ms(frame), 33)
        self.assertEqual(explicit_fps_sink._wait_ms(frame), 40)
        self.assertEqual(wait_sink._wait_ms(frame), 7)

    def test_displaysink_q_requests_pipeline_stop(self) -> None:
        frame = packet(np.zeros((4, 5, 3), dtype=np.uint8))
        context = PipelineContext()
        sink = DisplaySink("display", {"wait_ms": 1})
        sink.start(context)

        with (
            patch("src.sinks.displaysink.cv2.imshow"),
            patch("src.sinks.displaysink.cv2.waitKey", return_value=ord("q")),
        ):
            sink.process({"in": frame})

        self.assertTrue(context.stop_requested)

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
                    ElementSpec("eq", "hist_equalize", {"bins": 256}),
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
