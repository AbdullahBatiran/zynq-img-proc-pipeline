from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.cli import main as cli_main
from src.lib.cli_parse import parse_pipeline_expression
from src.lib.elements import PipelineContext
from src.lib.opencv_qt import configure_opencv_qt_environment
from src.lib.packets import FrameMetadata, FramePacket, new_packet_id
from src.lib.pipeline import ConnectionSpec, ElementSpec, Pipeline, PipelineSpec
from src.lib.registry import register_builtin_elements
from src.sinks.displaysink import DisplaySink
from src.sources.filesrc import infer_frame_format, normalize_decoded_frame
from src.transformers.bit_shift import BitShift
from src.transformers.bilateral import Bilateral
from src.transformers.bypass import Bypass
from src.transformers.combine import Combine
from src.transformers.debug import Debug
from src.transformers.dtype_convert import DtypeConvert
from src.transformers.fan_out import FanOut
from src.transformers.gaussian import Gaussian
from src.transformers.hist_equalize import HistEqualize
from src.transformers.laplacian_sharp import LaplacianSharp
from src.transformers.linear_scale import LinearScale
from src.transformers.median import Median
from src.transformers.mono_to_color import MonoToColor
from src.transformers.resize import Resize
from src.transformers.text_overlay import TextOverlay
from src.transformers.unsharp import Unsharp


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


def _changed_center(data: np.ndarray) -> tuple[int, int]:
    mask = np.any(data != 0, axis=2) if data.ndim == 3 else data != 0
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("frame has no changed pixels")
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    return ((int(min_x) + int(max_x)) // 2, (int(min_y) + int(max_y)) // 2)


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
        self.assertEqual(result.metadata.extra["hist_output_max"], 65535)

    def test_hist_equalize_uint16_output_bits_14_keeps_container_depth(self) -> None:
        transform = HistEqualize("eq", {"output-bits": 14})
        frame = np.linspace(0, 65535, 64, dtype=np.uint16).reshape((8, 8))
        result = transform.process({"in": packet(frame, fmt="gray")})["out"][0]

        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.metadata.depth, 16)
        self.assertLessEqual(int(result.data.max()), 16383)
        self.assertEqual(result.metadata.extra["hist_bins"], 16384)
        self.assertEqual(result.metadata.extra["hist_output_max"], 16383)
        self.assertEqual(result.metadata.extra["hist_output_bits"], 14)

    def test_hist_equalize_uint16_output_bits_12(self) -> None:
        transform = HistEqualize("eq", {"output_bits": 12})
        frame = np.linspace(0, 65535, 64, dtype=np.uint16).reshape((8, 8))
        result = transform.process({"in": packet(frame, fmt="gray")})["out"][0]

        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.metadata.depth, 16)
        self.assertLessEqual(int(result.data.max()), 4095)
        self.assertEqual(result.metadata.extra["hist_bins"], 4096)
        self.assertEqual(result.metadata.extra["hist_output_max"], 4095)
        self.assertEqual(result.metadata.extra["hist_output_bits"], 12)

    def test_hist_equalize_output_max(self) -> None:
        transform = HistEqualize("eq", {"output-max": 1000})
        frame = np.array([[0, 100, 1000], [2000, 3000, 5000]], dtype=np.uint16)
        result = transform.process({"in": packet(frame, fmt="gray")})["out"][0]

        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.metadata.depth, 16)
        self.assertLessEqual(int(result.data.max()), 1000)
        self.assertEqual(result.metadata.extra["hist_bins"], 1001)
        self.assertEqual(result.metadata.extra["hist_output_max"], 1000)
        self.assertNotIn("hist_output_bits", result.metadata.extra)

    def test_hist_equalize_rejects_invalid_output_range_params(self) -> None:
        for params in (
            {"output-bits": 0},
            {"output-max": -1},
            {"output-bits": 14, "output-max": 16383},
            {"output_bits": 14, "output-bits": 12},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                HistEqualize("eq", params)

        with self.assertRaises(ValueError):
            HistEqualize("eq", {"output-bits": 17}).process(
                {"in": packet(np.zeros((2, 2), dtype=np.uint16), fmt="gray")}
            )
        with self.assertRaises(ValueError):
            HistEqualize("eq", {"output-bits": 9}).process(
                {"in": packet(np.zeros((2, 2), dtype=np.uint8), fmt="gray")}
            )
        with self.assertRaises(ValueError):
            HistEqualize("eq", {"output-max": 65536}).process(
                {"in": packet(np.zeros((2, 2), dtype=np.uint16), fmt="gray")}
            )

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

    def test_filters_preserve_dtype_depth_shape_and_metadata(self) -> None:
        cases = [
            (Unsharp, "unsharp", {"amount": 0.5, "kernel-size": 3}),
            (Median, "median", {"kernel-size": 3}),
            (Gaussian, "gaussian", {"kernel-size": 3, "sigma-x": 0.0}),
            (
                Bilateral,
                "bilateral",
                {"diameter": 3, "sigma-color": 25.0, "sigma-space": 3.0},
            ),
            (
                LaplacianSharp,
                "laplacian-sharp",
                {"amount": 0.25, "kernel-size": 3, "iterations": 2},
            ),
        ]
        frames = [
            np.arange(25, dtype=np.uint8).reshape((5, 5)),
            (np.arange(25, dtype=np.uint16).reshape((5, 5)) * 100),
        ]

        for transform_cls, filter_name, params in cases:
            for frame in frames:
                with self.subTest(filter=filter_name, dtype=frame.dtype):
                    source = packet(frame, fmt="gray")
                    transform = transform_cls("f", params)
                    result = transform.process({"in": source})["out"][0]

                    self.assertEqual(result.data.dtype, frame.dtype)
                    self.assertEqual(result.data.shape, frame.shape)
                    self.assertEqual(result.metadata.depth, source.metadata.depth)
                    self.assertEqual(result.metadata.channels, source.metadata.channels)
                    self.assertEqual(result.metadata.extra["filtered_by"], "f")
                    self.assertEqual(result.metadata.extra["filter_name"], filter_name)
                    filter_params = result.metadata.extra["filter_params"]
                    for key, value in params.items():
                        self.assertEqual(filter_params[key], value)
                    self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_filters_preserve_three_channel_shape(self) -> None:
        frame = (np.arange(5 * 5 * 3, dtype=np.uint16).reshape((5, 5, 3)) * 100)
        cases = [
            Unsharp("f", {"kernel-size": 3}),
            Median("f", {"kernel-size": 3}),
            Gaussian("f", {"kernel-size": 3}),
            Bilateral("f", {"diameter": 3, "sigma-color": 100.0}),
            LaplacianSharp("f", {"kernel-size": 3}),
        ]

        for transform in cases:
            with self.subTest(filter=transform.type_name):
                result = transform.process({"in": packet(frame, fmt="rgb")})["out"][0]
                self.assertEqual(result.data.dtype, np.uint16)
                self.assertEqual(result.data.shape, frame.shape)
                self.assertEqual(result.metadata.channels, 3)
                self.assertEqual(result.metadata.format, "rgb")

    def test_filter_aliases_and_hyphenated_params(self) -> None:
        source = packet(np.arange(25, dtype=np.uint8).reshape((5, 5)), fmt="gray")
        transforms = [
            Unsharp("f", {"kernel_size": 3}),
            Gaussian("f", {"kernel_size": 3, "sigma_x": 1.0, "sigma_y": 1.0}),
            Bilateral("f", {"sigma_color": 20.0, "sigma_space": 2.0}),
            LaplacianSharp("f", {"kernel_size": 3}),
        ]

        for transform in transforms:
            with self.subTest(filter=transform.type_name):
                result = transform.process({"in": source})["out"][0]
                self.assertEqual(result.data.dtype, np.uint8)

    def test_filter_rejects_invalid_parameters(self) -> None:
        invalid_configurations = [
            (Unsharp, {"amount": -0.1}),
            (Unsharp, {"sigma": -1.0}),
            (Unsharp, {"kernel-size": 2}),
            (Median, {"kernel-size": 0}),
            (Median, {"kernel-size": 4}),
            (Gaussian, {"kernel-size": 2}),
            (Gaussian, {"sigma-x": -1.0}),
            (Gaussian, {"sigma-y": -1.0}),
            (Bilateral, {"diameter": 0}),
            (Bilateral, {"sigma-color": -1.0}),
            (Bilateral, {"sigma-space": -1.0}),
            (LaplacianSharp, {"amount": -0.1}),
            (LaplacianSharp, {"kernel-size": 2}),
            (LaplacianSharp, {"iterations": 0}),
            (LaplacianSharp, {"mode": "wrong"}),
            (LaplacianSharp, {"scale": -1.0}),
        ]
        for transform_cls, params in invalid_configurations:
            with self.subTest(transform=transform_cls.__name__, params=params):
                with self.assertRaises(ValueError):
                    transform_cls("f", params)

        with self.assertRaises(ValueError):
            Median("f", {"kernel-size": 7}).process(
                {"in": packet(np.zeros((5, 5), dtype=np.uint16), fmt="gray")}
            )

    def test_laplacian_sharp_iterations_change_output(self) -> None:
        frame = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 100, 100, 100, 0],
                [0, 100, 200, 100, 0],
                [0, 100, 100, 100, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        once = LaplacianSharp("lap", {"amount": 0.25, "iterations": 1}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]
        twice = LaplacianSharp("lap", {"amount": 0.25, "iterations": 2}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]

        self.assertEqual(once.data.dtype, np.uint16)
        self.assertEqual(twice.data.dtype, np.uint16)
        self.assertEqual(twice.metadata.extra["filter_params"]["iterations"], 2)
        self.assertFalse(np.array_equal(once.data, twice.data))

    def test_debug_default_prints_once_and_passes_same_packet(self) -> None:
        transform = Debug("dbg", {})
        source = packet(np.array([[1, 2], [3, 4]], dtype=np.uint8), fmt="gray")
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            first = transform.process({"in": source})["out"][0]
            second = transform.process({"in": source})["out"][0]

        output = stdout.getvalue()
        self.assertIs(first, source)
        self.assertIs(second, source)
        self.assertEqual(output.count("[debug dbg]"), 1)
        self.assertIn("shape=(2, 2)", output)
        self.assertIn("dtype=uint8", output)
        self.assertIn("min=1", output)
        self.assertIn("max=4", output)
        self.assertIn("mean=2.5", output)
        self.assertIn("std=1.11803", output)
        self.assertIn("median=2.5", output)

    def test_debug_disabled_suppresses_output_and_passes_packet(self) -> None:
        transform = Debug("dbg", {"enabled": False})
        source = packet(np.zeros((2, 2), dtype=np.uint8), fmt="gray")
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            result = transform.process({"in": source})["out"][0]

        self.assertIs(result, source)
        self.assertEqual(stdout.getvalue(), "")

    def test_debug_every_seconds_uses_packet_pts(self) -> None:
        transform = Debug("dbg", {"every-seconds": 1.0})
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            for index in (0, 15, 30, 45, 60):
                transform.process(
                    {"in": packet(np.zeros((1, 1), dtype=np.uint8), index=index)}
                )

        output = stdout.getvalue()
        self.assertIn("index=0 pts=0.000000", output)
        self.assertNotIn("index=15 pts=0.500000", output)
        self.assertIn("index=30 pts=1.000000", output)
        self.assertNotIn("index=45 pts=1.500000", output)
        self.assertIn("index=60 pts=2.000000", output)

    def test_debug_every_frames_prints_expected_indexes(self) -> None:
        transform = Debug("dbg", {"every-frames": 2})
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            for index in range(5):
                transform.process(
                    {"in": packet(np.zeros((1, 1), dtype=np.uint8), index=index)}
                )

        output = stdout.getvalue()
        self.assertIn("index=0", output)
        self.assertNotIn("index=1", output)
        self.assertIn("index=2", output)
        self.assertNotIn("index=3", output)
        self.assertIn("index=4", output)

    def test_debug_field_toggles_include_and_exclude_output(self) -> None:
        transform = Debug(
            "dbg",
            {
                "show_shape": False,
                "show-dtype": True,
                "show-min": True,
                "show-max": False,
                "show-mean": False,
                "show-std": False,
                "show-median": False,
                "show-preview": True,
                "preview-rows": 1,
                "preview-cols": 2,
                "preview-mode": "top-left",
            },
        )
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            transform.process(
                {"in": packet(np.array([[1, 2], [3, 4]], dtype=np.uint8), fmt="gray")}
            )

        output = stdout.getvalue()
        self.assertNotIn("shape=", output)
        self.assertIn("dtype=uint8", output)
        self.assertIn("min=1", output)
        self.assertNotIn("max=", output)
        self.assertNotIn("mean=", output)
        self.assertNotIn("std=", output)
        self.assertNotIn("median=", output)
        self.assertIn("preview top-left", output)
        self.assertNotIn("preview center", output)

    def test_debug_preview_both_includes_top_left_and_center(self) -> None:
        transform = Debug(
            "dbg",
            {"show-preview": True, "preview-rows": 2, "preview-cols": 2},
        )
        stdout = io.StringIO()
        frame = np.arange(16, dtype=np.uint8).reshape((4, 4))

        with contextlib.redirect_stdout(stdout):
            transform.process({"in": packet(frame, fmt="gray")})

        output = stdout.getvalue()
        self.assertIn("preview top-left rows=2 cols=2:", output)
        self.assertIn("[[0 1]\n [4 5]]", output)
        self.assertIn("preview center rows=2 cols=2:", output)
        self.assertIn("[[ 5  6]\n [ 9 10]]", output)

    def test_debug_supports_stdout_stderr_and_shared_file_output(self) -> None:
        stdout_transform = Debug("out", {"label": "stdout-label"})
        stderr_transform = Debug("err", {"stream": "stderr", "label": "stderr-label"})
        frame = packet(np.zeros((1, 1), dtype=np.uint8))
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            stdout_transform.process({"in": frame})
            stderr_transform.process({"in": frame})

        self.assertIn("[debug stdout-label]", stdout.getvalue())
        self.assertIn("[debug stderr-label]", stderr.getvalue())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "debug.log"
            first = Debug("a", {"stream": "file", "path": str(path), "label": "a"})
            second = Debug("b", {"stream": "file", "path": str(path), "label": "b"})
            first.process({"in": frame})
            second.process({"in": frame})
            first.stop()
            second.stop()

            output = path.read_text(encoding="utf-8")
            self.assertIn("[debug a]", output)
            self.assertIn("[debug b]", output)

    def test_debug_rejects_invalid_parameters(self) -> None:
        invalid_params = [
            {"every-seconds": 1, "every-frames": 1},
            {"every-seconds": 0},
            {"every-frames": 0},
            {"stream": "missing"},
            {"stream": "file"},
            {"preview-rows": 0},
            {"preview-cols": 0},
            {"preview-mode": "corner"},
            {"every_seconds": 1, "every-seconds": 2},
        ]
        for params in invalid_params:
            with self.subTest(params=params), self.assertRaises(ValueError):
                Debug("dbg", params)

    def test_combine_horizontal_three_inputs_preserves_parents(self) -> None:
        left = packet(np.zeros((4, 5, 3), dtype=np.uint8), stream_id="l")
        middle = packet(np.ones((4, 6, 3), dtype=np.uint8), stream_id="m")
        right = packet(np.full((4, 7, 3), 2, dtype=np.uint8), stream_id="r")
        transform = Combine("c", {"mode": "horizontal"})
        result = transform.process({"in0": left, "in1": middle, "in2": right})["out"][0]
        self.assertEqual(result.metadata.width, 18)
        self.assertEqual(result.metadata.height, 4)
        self.assertIn(left.metadata.packet_id, result.metadata.parents)
        self.assertIn(middle.metadata.packet_id, result.metadata.parents)
        self.assertIn(right.metadata.packet_id, result.metadata.parents)
        self.assertEqual(result.metadata.extra["input_ports"], ["in0", "in1", "in2"])

    def test_combine_vertical_three_inputs(self) -> None:
        top = packet(np.zeros((4, 5), dtype=np.uint8), stream_id="t", fmt="gray")
        middle = packet(np.ones((6, 5), dtype=np.uint8), stream_id="m", fmt="gray")
        bottom = packet(np.full((7, 5), 2, dtype=np.uint8), stream_id="b", fmt="gray")
        transform = Combine("c", {"mode": "vertical"})
        result = transform.process({"in0": top, "in1": middle, "in2": bottom})["out"][0]
        self.assertEqual(result.data.shape, (17, 5))
        self.assertEqual(result.metadata.width, 5)
        self.assertEqual(result.metadata.height, 17)
        self.assertEqual(result.metadata.extra["combine_mode"], "vertical")

    def test_combine_grid_full_inputs(self) -> None:
        frames = {
            f"in{index}": packet(
                np.full((2, 3), index, dtype=np.uint8),
                stream_id=f"s{index}",
                fmt="gray",
            )
            for index in range(4)
        }
        transform = Combine("c", {"mode": "grid", "rows": 2, "cols": 2})
        result = transform.process(frames)["out"][0]

        self.assertEqual(result.data.tolist(), [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [2, 2, 2, 3, 3, 3],
            [2, 2, 2, 3, 3, 3],
        ])
        self.assertEqual(result.metadata.width, 6)
        self.assertEqual(result.metadata.height, 4)
        self.assertEqual(result.metadata.extra["grid_rows"], 2)
        self.assertEqual(result.metadata.extra["grid_cols"], 2)
        self.assertEqual(result.metadata.extra["missing_input_ports"], [])

    def test_combine_grid_fills_unconnected_cells_black(self) -> None:
        frames = {
            "in0": packet(np.full((2, 2), 1, dtype=np.uint8), stream_id="a", fmt="gray"),
            "in2": packet(np.full((2, 2), 2, dtype=np.uint8), stream_id="b", fmt="gray"),
        }
        transform = Combine("c", {"mode": "grid", "rows": 2, "cols": 2})
        result = transform.process(frames)["out"][0]

        self.assertEqual(result.data.tolist(), [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [2, 2, 0, 0],
            [2, 2, 0, 0],
        ])
        self.assertEqual(result.metadata.extra["missing_input_ports"], ["in1", "in3"])

    def test_combine_rejects_depth_mismatch(self) -> None:
        left = packet(np.zeros((4, 5, 3), dtype=np.uint8))
        right = packet(np.zeros((4, 5, 3), dtype=np.uint16))
        transform = Combine("c", {"mode": "horizontal"})
        with self.assertRaises(ValueError):
            transform.process({"in0": left, "in1": right})

    def test_combine_rejects_incompatible_inputs(self) -> None:
        base = packet(np.zeros((4, 5), dtype=np.uint8), fmt="gray")
        cases = [
            (
                Combine("c", {"mode": "horizontal"}),
                {
                    "in0": base,
                    "in1": packet(np.zeros((4, 5), dtype=np.uint8), fmt="bgr"),
                },
            ),
            (
                Combine("c", {"mode": "horizontal"}),
                {
                    "in0": base,
                    "in1": packet(np.zeros((4, 5, 3), dtype=np.uint8), fmt="gray"),
                },
            ),
            (
                Combine("c", {"mode": "horizontal"}),
                {
                    "in0": base,
                    "in1": packet(
                        np.zeros((4, 5), dtype=np.uint8),
                        fmt="gray",
                        index=1,
                    ),
                },
            ),
            (
                Combine("c", {"mode": "horizontal"}),
                {
                    "in0": base,
                    "in1": packet(np.zeros((3, 5), dtype=np.uint8), fmt="gray"),
                },
            ),
            (
                Combine("c", {"mode": "vertical"}),
                {
                    "in0": base,
                    "in1": packet(np.zeros((4, 6), dtype=np.uint8), fmt="gray"),
                },
            ),
            (
                Combine("c", {"mode": "grid", "rows": 1, "cols": 2}),
                {
                    "in0": base,
                    "in1": packet(np.zeros((4, 6), dtype=np.uint8), fmt="gray"),
                },
            ),
        ]
        for transform, inputs in cases:
            with self.subTest(mode=transform.mode), self.assertRaises(ValueError):
                transform.process(inputs)

    def test_combine_rejects_invalid_grid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            Combine("c", {"mode": "grid"})
        with self.assertRaises(ValueError):
            Combine("c", {"mode": "overlay"})
        with self.assertRaises(ValueError):
            Combine("c", {"mode": "grid", "rows": 2, "cols": 2}).process(
                {"in4": packet(np.zeros((2, 2), dtype=np.uint8), fmt="gray")}
            )

    def test_bit_shift_right_uint8_gray(self) -> None:
        frame = np.array([[0, 1, 2, 4], [8, 16, 32, 255]], dtype=np.uint8)
        source = packet(frame, fmt="gray")
        result = BitShift("shift", {"bits": 2}).process({"in": source})["out"][0]

        self.assertEqual(result.data.tolist(), [[0, 0, 0, 1], [2, 4, 8, 63]])
        self.assertEqual(result.data.dtype, np.uint8)
        self.assertEqual(result.metadata.format, "gray")
        self.assertEqual(result.metadata.depth, 8)
        self.assertEqual(result.metadata.extra["bit_shifted_by"], "shift")
        self.assertEqual(result.metadata.extra["bit_shift_bits"], 2)
        self.assertEqual(result.metadata.extra["bit_shift_direction"], "right")
        self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_bit_shift_left_uint8_wraps_like_numpy(self) -> None:
        frame = np.array([[1, 64, 128, 255]], dtype=np.uint8)
        result = BitShift("shift", {"bits": 1, "direction": "left"}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]

        self.assertEqual(result.data.tolist(), np.left_shift(frame, 1).tolist())
        self.assertEqual(int(result.data[0, 2]), 0)
        self.assertEqual(int(result.data[0, 3]), 254)

    def test_bit_shift_uint16_color_preserves_shape_dtype_and_depth(self) -> None:
        frame = np.array(
            [[[1, 2, 4], [8, 16, 32]], [[64, 128, 256], [512, 1024, 2048]]],
            dtype=np.uint16,
        )
        source = packet(frame, fmt="rgb")
        result = BitShift("shift", {"bits": 3, "direction": "left"}).process(
            {"in": source}
        )["out"][0]

        self.assertEqual(result.data.tolist(), np.left_shift(frame, 3).tolist())
        self.assertEqual(result.data.shape, frame.shape)
        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.metadata.format, "rgb")
        self.assertEqual(result.metadata.channels, 3)
        self.assertEqual(result.metadata.depth, 16)

    def test_bit_shift_zero_bits_derives_equivalent_packet(self) -> None:
        source = packet(np.array([[1, 2], [3, 4]], dtype=np.uint8), fmt="gray")
        result = BitShift("shift", {"bits": 0}).process({"in": source})["out"][0]

        np.testing.assert_array_equal(result.data, source.data)
        self.assertIsNot(result, source)
        self.assertNotEqual(result.metadata.packet_id, source.metadata.packet_id)
        self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_bit_shift_rejects_invalid_parameters_and_frames(self) -> None:
        for params in ({"bits": -1}, {"bits": 1, "direction": "up"}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                BitShift("shift", params)

        transform = BitShift("shift", {"bits": 1})
        with self.assertRaises(ValueError):
            transform.process(
                {"in": packet(np.zeros((2, 2), dtype=np.float32), fmt="gray")}
            )
        with self.assertRaises(ValueError):
            transform.process(
                {"in": packet(np.zeros((2, 2), dtype=np.uint8), fmt="mono")}
            )
        with self.assertRaises(ValueError):
            transform.process(
                {"in": packet(np.zeros((2, 2, 4), dtype=np.uint8), fmt="rgb")}
            )

    def test_dtype_convert_uint8_to_uint16_preserves_values(self) -> None:
        frame = np.array([[0, 1, 255]], dtype=np.uint8)
        source = packet(frame, fmt="gray")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            result = DtypeConvert("dtype", {"dtype": "uint16"}).process(
                {"in": source}
            )["out"][0]

        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.data.tolist(), [[0, 1, 255]])
        self.assertEqual(result.metadata.depth, 16)
        self.assertEqual(result.metadata.format, "gray")
        self.assertEqual(result.metadata.extra["dtype_converted_by"], "dtype")
        self.assertEqual(result.metadata.extra["dtype_convert_input_dtype"], "uint8")
        self.assertEqual(result.metadata.extra["dtype_convert_output_dtype"], "uint16")
        self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_dtype_convert_uint16_to_uint8_clips_and_warns(self) -> None:
        frame = np.array([[0, 255, 256, 1000]], dtype=np.uint16)
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            result = DtypeConvert("dtype", {"dtype": "uint8"}).process(
                {"in": packet(frame, fmt="gray")}
            )["out"][0]

        self.assertEqual(result.data.dtype, np.uint8)
        self.assertEqual(result.data.tolist(), [[0, 255, 255, 255]])
        self.assertIn("\033[33m", stderr.getvalue())
        self.assertIn(
            "Warning: dtype-convert clipping values above 255",
            stderr.getvalue(),
        )
        self.assertIn("first clipped value=256 at row=0, col=2", stderr.getvalue())
        self.assertIn("\033[0m", stderr.getvalue())

    def test_dtype_convert_uint32_color_to_uint16_clips(self) -> None:
        frame = np.array(
            [[[0, 1, 65535], [65536, 70000, 100000]]],
            dtype=np.uint32,
        )
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            result = DtypeConvert("dtype", {"dtype": "uint16"}).process(
                {"in": packet(frame, fmt="bgr")}
            )["out"][0]

        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(
            result.data.tolist(),
            [[[0, 1, 65535], [65535, 65535, 65535]]],
        )
        self.assertEqual(result.metadata.depth, 16)
        self.assertEqual(result.metadata.channels, 3)
        self.assertIn("\033[33m", stderr.getvalue())
        self.assertIn(
            "Warning: dtype-convert clipping values above 65535",
            stderr.getvalue(),
        )
        self.assertIn(
            "first clipped value=65536 at row=0, col=1, channel=0",
            stderr.getvalue(),
        )
        self.assertIn("\033[0m", stderr.getvalue())

    def test_dtype_convert_rejects_invalid_parameters_and_frames(self) -> None:
        with self.assertRaises(ValueError):
            DtypeConvert("dtype", {"dtype": "float32"})

        transform = DtypeConvert("dtype", {"dtype": "uint8"})
        with self.assertRaises(ValueError):
            transform.process(
                {"in": packet(np.zeros((2, 2), dtype=np.float32), fmt="gray")}
            )
        with self.assertRaises(ValueError):
            transform.process(
                {"in": packet(np.zeros((2, 2), dtype=np.uint8), fmt="mono")}
            )
        with self.assertRaises(ValueError):
            transform.process(
                {"in": packet(np.zeros((2, 2, 4), dtype=np.uint8), fmt="rgb")}
            )

    def test_mono_to_color_defaults_to_bgr_from_2d_gray(self) -> None:
        frame = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        source = packet(frame, fmt="gray")
        result = MonoToColor("color", {}).process({"in": source})["out"][0]

        expected = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
        np.testing.assert_array_equal(result.data, expected)
        self.assertEqual(result.data.dtype, np.uint8)
        self.assertEqual(result.metadata.format, "bgr")
        self.assertEqual(result.metadata.channels, 3)
        self.assertEqual(result.metadata.width, 2)
        self.assertEqual(result.metadata.height, 2)
        self.assertEqual(result.metadata.extra["mono_to_color_by"], "color")
        self.assertEqual(result.metadata.extra["mono_to_color_format"], "bgr")
        self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_mono_to_color_supports_rgb_and_hxwx1_input(self) -> None:
        frame = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
        result = MonoToColor("color", {"format": "rgb"}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]

        self.assertEqual(result.metadata.format, "rgb")
        self.assertEqual(result.metadata.channels, 3)
        self.assertEqual(result.data.tolist(), [
            [[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
        ])

    def test_mono_to_color_preserves_uint16_values(self) -> None:
        frame = np.array([[0, 1024], [4096, 65535]], dtype=np.uint16)
        result = MonoToColor("color", {}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]

        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.metadata.depth, 16)
        np.testing.assert_array_equal(result.data[:, :, 0], frame)
        np.testing.assert_array_equal(result.data[:, :, 1], frame)
        np.testing.assert_array_equal(result.data[:, :, 2], frame)

    def test_mono_to_color_rejects_invalid_parameters_and_frames(self) -> None:
        with self.assertRaises(ValueError):
            MonoToColor("color", {"format": "gray"})

        transform = MonoToColor("color", {})
        with self.assertRaises(ValueError):
            transform.process(
                {"in": packet(np.zeros((2, 2, 3), dtype=np.uint8), fmt="bgr")}
            )
        with self.assertRaises(ValueError):
            transform.process(
                {"in": packet(np.zeros((2, 2, 3), dtype=np.uint8), fmt="gray")}
            )
        with self.assertRaises(ValueError):
            transform.process(
                {"in": packet(np.zeros((2, 2), dtype=np.float32), fmt="gray")}
            )

    def test_bypass_passes_same_packet_without_parameters(self) -> None:
        transform = Bypass("bypass", {})
        source = packet(np.zeros((2, 2), dtype=np.uint32), fmt="custom")

        result = transform.process({"in": source})["out"][0]

        self.assertIs(result, source)
        self.assertEqual(transform.params, {})

    def test_text_overlay_draws_bgr_text_and_preserves_metadata(self) -> None:
        frame = np.zeros((80, 160, 3), dtype=np.uint8)
        source = packet(frame)
        transform = TextOverlay(
            "txt",
            {
                "text": "A",
                "color": "red",
                "position": "top-left",
                "line-type": "8",
                "thickness": 2,
            },
        )

        result = transform.process({"in": source})["out"][0]

        self.assertEqual(result.data.shape, frame.shape)
        self.assertEqual(result.data.dtype, np.uint8)
        self.assertEqual(result.metadata.width, source.metadata.width)
        self.assertEqual(result.metadata.height, source.metadata.height)
        self.assertEqual(result.metadata.format, "bgr")
        self.assertIn(source.metadata.packet_id, result.metadata.parents)
        self.assertEqual(result.metadata.extra["text_overlay_by"], "txt")
        self.assertEqual(result.metadata.extra["text_overlay_text"], "A")
        self.assertEqual(result.metadata.extra["text_overlay_color"], (255, 0, 0))
        self.assertGreater(int(result.data[:, :, 2].max()), 0)
        self.assertEqual(int(result.data[:, :, 0].max()), 0)
        self.assertEqual(int(result.data[:, :, 1].max()), 0)
        self.assertEqual(int(source.data.max()), 0)

    def test_text_overlay_handles_gray_rgb_and_bgr_color_ordering(self) -> None:
        params = {
            "text": "I",
            "color": "red",
            "position": "top-left",
            "line-type": "8",
            "thickness": 2,
        }

        gray = TextOverlay("txt", params).process(
            {"in": packet(np.zeros((60, 80), dtype=np.uint8), fmt="gray")}
        )["out"][0]
        bgr = TextOverlay("txt", params).process(
            {"in": packet(np.zeros((60, 80, 3), dtype=np.uint8), fmt="bgr")}
        )["out"][0]
        rgb = TextOverlay("txt", params).process(
            {"in": packet(np.zeros((60, 80, 3), dtype=np.uint8), fmt="rgb")}
        )["out"][0]

        self.assertEqual(int(gray.data.max()), 76)
        self.assertGreater(int(bgr.data[:, :, 2].max()), 0)
        self.assertEqual(int(bgr.data[:, :, 0].max()), 0)
        self.assertGreater(int(rgb.data[:, :, 0].max()), 0)
        self.assertEqual(int(rgb.data[:, :, 2].max()), 0)

    def test_text_overlay_scales_uint16_colors(self) -> None:
        transform = TextOverlay(
            "txt",
            {
                "text": "A",
                "color": "#00FF00",
                "position": "top-left",
                "line-type": "8",
            },
        )
        source = packet(np.zeros((80, 160, 3), dtype=np.uint16), fmt="bgr")

        result = transform.process({"in": source})["out"][0]

        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.metadata.depth, 16)
        self.assertEqual(int(result.data[:, :, 1].max()), 65535)
        self.assertEqual(int(result.data[:, :, 0].max()), 0)
        self.assertEqual(int(result.data[:, :, 2].max()), 0)

    def test_text_overlay_anchor_positions(self) -> None:
        cases = {
            "top-left": (0, 45, 0, 45),
            "center": (55, 105, 35, 85),
            "bottom-right": (115, 159, 80, 119),
        }
        for position, (min_x, max_x, min_y, max_y) in cases.items():
            with self.subTest(position=position):
                transform = TextOverlay(
                    "txt",
                    {
                        "text": "A",
                        "position": position,
                        "line-type": "8",
                    },
                )
                result = transform.process(
                    {"in": packet(np.zeros((120, 160), dtype=np.uint8), fmt="gray")}
                )["out"][0]
                center_x, center_y = _changed_center(result.data)

                self.assertGreaterEqual(center_x, min_x)
                self.assertLessEqual(center_x, max_x)
                self.assertGreaterEqual(center_y, min_y)
                self.assertLessEqual(center_y, max_y)

    def test_text_overlay_rejects_invalid_parameters_and_frames(self) -> None:
        invalid_params = [
            {"text": ""},
            {"text": "A", "color": "wrong"},
            {"text": "A", "color": "256,0,0"},
            {"text": "A", "position": "corner"},
            {"text": "A", "font-size": 0},
            {"text": "A", "thickness": 0},
            {"text": "A", "font": "truetype"},
            {"text": "A", "line-type": "16"},
        ]
        for params in invalid_params:
            with self.subTest(params=params), self.assertRaises(ValueError):
                TextOverlay("txt", params)

        transform = TextOverlay("txt", {"text": "A"})
        with self.assertRaises(ValueError):
            transform.process(
                {"in": packet(np.zeros((8, 8), dtype=np.float32), fmt="gray")}
            )
        with self.assertRaises(ValueError):
            transform.process(
                {"in": packet(np.zeros((8, 8, 4), dtype=np.uint8), fmt="rgba")}
            )

    def test_fan_out_replicates_packet_to_requested_outputs(self) -> None:
        source = packet(np.zeros((4, 5), dtype=np.uint8), fmt="gray")
        transform = FanOut("f", {"outputs": 3})
        result = transform.process({"in": source})

        self.assertEqual(set(result), {"out0", "out1", "out2"})
        self.assertIs(result["out0"][0], source)
        self.assertIs(result["out1"][0], source)
        self.assertIs(result["out2"][0], source)

    def test_fan_out_uses_connected_dynamic_outputs(self) -> None:
        source = packet(np.zeros((4, 5), dtype=np.uint8), fmt="gray")
        transform = FanOut("f", {})
        transform.configure_connected_output_ports({"out2", "out0"})
        result = transform.process({"in": source})

        self.assertEqual(list(result), ["out0", "out2"])
        self.assertIs(result["out0"][0], source)
        self.assertIs(result["out2"][0], source)

    def test_fan_out_rejects_invalid_params_and_ports(self) -> None:
        with self.assertRaises(ValueError):
            FanOut("f", {"outputs": 0})
        with self.assertRaises(ValueError):
            FanOut("f", {}).configure_connected_output_ports({"output0"})
        with self.assertRaises(ValueError):
            FanOut("f", {"outputs": 2}).configure_connected_output_ports({"out2"})

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

    def test_cli_parser_allows_linear_expression_across_lines(self) -> None:
        spec = parse_pipeline_expression(
            """
            filesrc path=in.mp4
            ! resize width=4 height=4
            ! filesink path=out.mp4
            """
        )
        self.assertEqual(
            [element.type for element in spec.elements],
            ["filesrc", "resize", "filesink"],
        )
        self.assertEqual(len(spec.connections), 2)

    def test_cli_parser_accepts_hyphenated_linear_scale_params(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mp4 ! linear-scale otype=uint8 perc-up=0.01 "
            "! filesink path=out.mp4"
        )
        self.assertEqual(spec.elements[1].type, "linear-scale")
        self.assertEqual(spec.elements[1].params["perc-up"], 0.01)

    def test_cli_parser_accepts_hyphenated_debug_params(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mp4 ! debug every-seconds=1 show-preview=true "
            "! filesink path=out.mp4"
        )
        self.assertEqual(spec.elements[1].type, "debug")
        self.assertEqual(spec.elements[1].params["every-seconds"], 1)
        self.assertTrue(spec.elements[1].params["show-preview"])

    def test_cli_parser_accepts_hyphenated_filter_params(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mp4 ! Unsharp kernel-size=3 ! gaussian sigma-x=1.0 "
            "! bilateral sigma-color=25 sigma-space=3 ! laplacian-sharp "
            "kernel-size=3 iterations=2 ! filesink path=out.mp4"
        )

        self.assertEqual(spec.elements[1].type, "Unsharp")
        self.assertEqual(spec.elements[1].params["kernel-size"], 3)
        self.assertEqual(spec.elements[2].type, "gaussian")
        self.assertEqual(spec.elements[2].params["sigma-x"], 1.0)
        self.assertEqual(spec.elements[3].type, "bilateral")
        self.assertEqual(spec.elements[3].params["sigma-color"], 25)
        self.assertEqual(spec.elements[3].params["sigma-space"], 3)
        self.assertEqual(spec.elements[4].type, "laplacian-sharp")
        self.assertEqual(spec.elements[4].params["kernel-size"], 3)
        self.assertEqual(spec.elements[4].params["iterations"], 2)

    def test_cli_parser_accepts_text_overlay_quoted_text(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mp4 ! text-overlay text='Frame 1' color=red "
            "position=bottom-right font-size=0.8 ! filesink path=out.mp4"
        )

        self.assertEqual(spec.elements[1].type, "text-overlay")
        self.assertEqual(spec.elements[1].params["text"], "Frame 1")
        self.assertEqual(spec.elements[1].params["position"], "bottom-right")
        self.assertEqual(spec.elements[1].params["font-size"], 0.8)

    def test_cli_parser_accepts_bit_shift_and_mono_to_color(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mkv ! bit-shift bits=2 direction=left "
            "! mono-to-color format=rgb ! filesink path=out.mp4"
        )

        self.assertEqual(spec.elements[1].type, "bit-shift")
        self.assertEqual(spec.elements[1].params["bits"], 2)
        self.assertEqual(spec.elements[1].params["direction"], "left")
        self.assertEqual(spec.elements[2].type, "mono-to-color")
        self.assertEqual(spec.elements[2].params["format"], "rgb")

    def test_cli_parser_accepts_dtype_convert(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mkv ! dtype-convert dtype=uint16 ! filesink path=out.mp4"
        )

        self.assertEqual(spec.elements[1].type, "dtype-convert")
        self.assertEqual(spec.elements[1].params["dtype"], "uint16")

    def test_cli_parser_accepts_bypass(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mkv ! bypass ! filesink path=out.mp4"
        )

        self.assertEqual(spec.elements[1].type, "bypass")
        self.assertEqual(spec.elements[1].params, {})

    def test_cli_parser_named_graph(self) -> None:
        spec = parse_pipeline_expression(
            """
            filesrc name=a path=a.mp4 ! resize name=ra width=4 height=4
            filesrc name=b path=b.mp4 ! resize name=rb width=4 height=4
            ra.out ! combine.in0 name=c mode=horizontal
            rb.out ! c.in1
            c.out ! filesink path=out.mp4
            """
        )
        ids = {element.id for element in spec.elements}
        self.assertTrue({"a", "b", "ra", "rb", "c", "filesink"}.issubset(ids))
        self.assertIn(ConnectionSpec("rb", "out", "c", "in1"), spec.connections)

    def test_cli_parser_named_fan_out_branches(self) -> None:
        spec = parse_pipeline_expression(
            """
            filesrc name=src path=in.mp4 ! fan-out name=f
            f.out0 ! debug name=a enabled=false
            f.out1 ! debug name=b enabled=false
            """
        )
        ids = {element.id for element in spec.elements}
        self.assertTrue({"src", "f", "a", "b"}.issubset(ids))
        self.assertIn(ConnectionSpec("f", "out0", "a", "in"), spec.connections)
        self.assertIn(ConnectionSpec("f", "out1", "b", "in"), spec.connections)

    def test_pipeline_validation_accepts_dynamic_combine_ports(self) -> None:
        spec = PipelineSpec(
            elements=[
                ElementSpec("src_a", "filesrc", {"path": "a.mp4"}),
                ElementSpec("src_b", "filesrc", {"path": "b.mp4"}),
                ElementSpec("combo", "combine", {"mode": "horizontal"}),
                ElementSpec("out", "filesink", {"path": "out.mp4"}),
            ],
            connections=[
                ConnectionSpec("src_a", "out", "combo", "in0"),
                ConnectionSpec("src_b", "out", "combo", "in1"),
                ConnectionSpec("combo", "out", "out", "in"),
            ],
        )
        Pipeline.from_spec(spec)

    def test_pipeline_validation_rejects_invalid_dynamic_combine_port(self) -> None:
        spec = PipelineSpec(
            elements=[
                ElementSpec("src_a", "filesrc", {"path": "a.mp4"}),
                ElementSpec("combo", "combine", {"mode": "horizontal"}),
            ],
            connections=[ConnectionSpec("src_a", "out", "combo", "foo")],
        )
        with self.assertRaises(ValueError):
            Pipeline.from_spec(spec)

    def test_pipeline_validation_accepts_dynamic_fan_out_ports(self) -> None:
        spec = PipelineSpec(
            elements=[
                ElementSpec("src", "filesrc", {"path": "a.mp4"}),
                ElementSpec("fan", "fan-out", {}),
                ElementSpec("first", "debug", {"enabled": False}),
                ElementSpec("second", "debug", {"enabled": False}),
                ElementSpec("out", "filesink", {"path": "out.mp4"}),
            ],
            connections=[
                ConnectionSpec("src", "out", "fan", "in"),
                ConnectionSpec("fan", "out0", "first", "in"),
                ConnectionSpec("fan", "out1", "second", "in"),
                ConnectionSpec("first", "out", "out", "in"),
            ],
        )
        pipeline = Pipeline.from_spec(spec)
        self.assertEqual(
            pipeline.elements["fan"].connected_output_ports,
            {"out0", "out1"},
        )

    def test_pipeline_validation_rejects_invalid_dynamic_fan_out_port(self) -> None:
        spec = PipelineSpec(
            elements=[
                ElementSpec("src", "filesrc", {"path": "a.mp4"}),
                ElementSpec("fan", "fan-out", {}),
                ElementSpec("dbg", "debug", {"enabled": False}),
            ],
            connections=[
                ConnectionSpec("src", "out", "fan", "in"),
                ConnectionSpec("fan", "output0", "dbg", "in"),
            ],
        )
        with self.assertRaises(ValueError):
            Pipeline.from_spec(spec)

    def test_cli_run_loads_pipeline_script_with_placeholders(self) -> None:
        class DummyPipeline:
            def __init__(self) -> None:
                self.max_frames = None

            def run(self, max_frames=None) -> None:
                self.max_frames = max_frames

        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "pipeline.zpipe"
            script_path.write_text(
                """
                # Long pipelines can be split across lines.
                filesrc path=$1
                ! resize width=4 height=4
                ! filesink path=$2
                """,
                encoding="utf-8",
            )
            dummy = DummyPipeline()
            with patch("src.cli.Pipeline.from_spec", return_value=dummy) as from_spec:
                exit_code = cli_main(
                    [
                        "run",
                        "--file",
                        str(script_path),
                        "input video.mp4",
                        "out.mp4",
                        "--max-frames",
                        "7",
                    ]
                )

            spec = from_spec.call_args.args[0]
            self.assertEqual(exit_code, 0)
            self.assertEqual(dummy.max_frames, 7)
            self.assertEqual(spec.elements[0].params["path"], "input video.mp4")
            self.assertEqual(spec.elements[2].params["path"], "out.mp4")

    def test_cli_run_accepts_script_path_as_expression(self) -> None:
        class DummyPipeline:
            def run(self, max_frames=None) -> None:
                pass

        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "pipeline.zpipe"
            script_path.write_text(
                "filesrc path=$1 ! filesink path=$2",
                encoding="utf-8",
            )
            with patch(
                "src.cli.Pipeline.from_spec", return_value=DummyPipeline()
            ) as from_spec:
                exit_code = cli_main(["run", str(script_path), "in.mp4", "out.mp4"])

            spec = from_spec.call_args.args[0]
            self.assertEqual(exit_code, 0)
            self.assertEqual(spec.elements[0].params["path"], "in.mp4")
            self.assertEqual(spec.elements[1].params["path"], "out.mp4")

    def test_cli_run_reports_missing_script_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "pipeline.zpipe"
            script_path.write_text(
                "filesrc path=$1 ! filesink path=$2",
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = cli_main(["run", "--file", str(script_path), "in.mp4"])

            self.assertEqual(exit_code, 1)
            self.assertIn(
                "Missing value for pipeline script placeholder $2",
                stderr.getvalue(),
            )

    def test_cli_list_elements_groups_registered_elements(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["list-elements"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Registered elements (", output)
        self.assertIn("Sources", output)
        self.assertIn("Transformers", output)
        self.assertIn("Sinks", output)
        self.assertIn("Name     Subcategory  Description", output)
        self.assertIn("filesrc  File         Read video frames", output)
        self.assertIn("combine          Compose", output)
        self.assertIn("text-overlay     Compose", output)
        self.assertIn("mono-to-color    Color", output)
        self.assertIn("bypass           Control", output)
        self.assertIn("fan-out          Control", output)
        self.assertIn("hist_equalize    Contrast", output)
        self.assertIn("linear-scale     Contrast", output)
        self.assertIn("debug            Debug", output)
        self.assertIn("bilateral        Filter", output)
        self.assertIn("resize           Geometry", output)
        self.assertIn("bit-shift        Intensity", output)
        self.assertIn("dtype-convert    Intensity", output)
        self.assertIn("filesink     File", output)
        self.assertIn("displaysink  GUI", output)

    def test_cli_list_elements_verbose_shows_readable_element_blocks(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["list-elements", "--verbose"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Registered elements (", output)
        self.assertIn("Sources\n  File\n    filesrc", output)
        self.assertIn("Inputs: none", output)
        self.assertIn(
            "Parameters: path*, stream_id, source_id, format, depth, preserve_native",
            output,
        )
        self.assertIn("* required", output)
        self.assertIn("Transformers", output)
        self.assertIn("  Compose\n    combine", output)
        self.assertIn("    text-overlay", output)
        self.assertIn("Parameters: text*, color, position, x, y, font-size", output)
        self.assertIn("  Color\n    mono-to-color", output)
        self.assertIn("Parameters: format", output)
        self.assertIn("  Control\n    bypass", output)
        self.assertIn("    fan-out", output)
        self.assertIn("Outputs: outN", output)
        self.assertIn("Parameters: outputs", output)
        self.assertIn("Parameters: none", output)
        self.assertIn("  Contrast\n    hist_equalize", output)
        self.assertIn("    linear-scale", output)
        self.assertIn("  Debug\n    debug", output)
        self.assertIn("  Filter\n    bilateral", output)
        self.assertIn("Parameters: diameter, sigma-color, sigma-space", output)
        self.assertIn("Inputs: inN", output)
        self.assertIn("Parameters: mode, rows, cols, stream_id", output)
        self.assertIn("every-seconds, every-frames", output)
        self.assertIn("show-preview", output)
        self.assertIn("    gaussian", output)
        self.assertIn("Parameters: kernel-size, sigma-x, sigma-y", output)
        self.assertIn("Parameters: bins, output-bits, output-max", output)
        self.assertIn("    laplacian-sharp", output)
        self.assertIn(
            "Parameters: amount, kernel-size, iterations, mode, scale, delta",
            output,
        )
        self.assertIn("otype, omin, omax, min, max, perc, perc-down, perc-up", output)
        self.assertIn("    median", output)
        self.assertIn("    unsharp", output)
        self.assertIn("  Geometry\n    resize", output)
        self.assertIn("  Intensity\n    bit-shift", output)
        self.assertIn("Parameters: bits*, direction", output)
        self.assertIn("    dtype-convert", output)
        self.assertIn("Parameters: dtype*", output)
        self.assertIn("Sinks\n  File\n    filesink", output)
        self.assertIn("  GUI\n    displaysink", output)

    def test_cli_describe_linear_scale_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "linear-scale"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: linear-scale", output)
        self.assertIn("Subcategory: Contrast", output)
        self.assertIn("otype: str | optional", output)
        self.assertIn("perc-up: float | optional", output)
        self.assertIn("formats=[bgr, gray, rgb]", output)

    def test_cli_describe_hist_equalize_shows_output_range_params(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "hist_equalize"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: hist_equalize", output)
        self.assertIn("bins: int | optional", output)
        self.assertIn("output-bits: int | optional", output)
        self.assertIn("output-max: int | optional", output)

    def test_cli_describe_debug_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "debug"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: debug", output)
        self.assertIn("every-seconds: float | optional", output)
        self.assertIn("show-preview: bool | optional", output)
        self.assertIn("preview-mode: str | optional", output)

    def test_cli_describe_combine_shows_dynamic_inputs(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "combine"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: combine", output)
        self.assertIn("inN: FramePacket", output)
        self.assertIn("mode: str | optional", output)
        self.assertIn("choices=[grid, horizontal, vertical]", output)
        self.assertIn("rows: int | optional", output)
        self.assertIn("cols: int | optional", output)
        self.assertNotIn("overlay", output)
        self.assertNotIn("alpha", output)

    def test_cli_describe_text_overlay_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "text-overlay"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: text-overlay", output)
        self.assertIn("Subcategory: Compose", output)
        self.assertIn("text: str | required", output)
        self.assertIn("font-size: float | optional", output)
        self.assertIn("line-type: str | optional", output)
        self.assertIn("formats=[bgr, gray, rgb]", output)

    def test_cli_describe_bit_shift_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "bit-shift"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: bit-shift", output)
        self.assertIn("Subcategory: Intensity", output)
        self.assertIn("bits: int | required", output)
        self.assertIn("direction: str | optional", output)
        self.assertIn("choices=[left, right]", output)
        self.assertIn("formats=[bgr, gray, rgb]", output)

    def test_cli_describe_dtype_convert_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "dtype-convert"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: dtype-convert", output)
        self.assertIn("Subcategory: Intensity", output)
        self.assertIn("dtype: str | required", output)
        self.assertIn("choices=[uint8, uint16, uint32]", output)
        self.assertIn("depths=[8, 16, 32]", output)

    def test_cli_describe_mono_to_color_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "mono-to-color"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: mono-to-color", output)
        self.assertIn("Subcategory: Color", output)
        self.assertIn("format: str | optional", output)
        self.assertIn("choices=[bgr, rgb]", output)
        self.assertIn("formats=[gray]", output)
        self.assertIn("formats=[bgr, rgb]", output)

    def test_cli_describe_bypass_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "bypass"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: bypass", output)
        self.assertIn("Subcategory: Control", output)
        self.assertIn("Parameters:\n  none", output)
        self.assertIn("in: FramePacket", output)
        self.assertIn("out: FramePacket", output)

    def test_cli_describe_fan_out_shows_dynamic_outputs(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "fan-out"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: fan-out", output)
        self.assertIn("Subcategory: Control", output)
        self.assertIn("outputs: int | optional", output)
        self.assertIn("Input ports:", output)
        self.assertIn("in: FramePacket", output)
        self.assertIn("Output ports:", output)
        self.assertIn("outN: FramePacket", output)

    def test_cli_describe_filter_elements(self) -> None:
        expectations = {
            "unsharp": ["amount: float | optional", "kernel-size: int | optional"],
            "median": ["kernel-size: int | optional"],
            "gaussian": ["sigma-x: float | optional", "sigma-y: float | optional"],
            "bilateral": [
                "diameter: int | optional",
                "sigma-color: float | optional",
                "sigma-space: float | optional",
            ],
            "laplacian-sharp": [
                "amount: float | optional",
                "iterations: int | optional",
                "mode: str | optional",
            ],
        }

        for element_name, expected_lines in expectations.items():
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(["describe", element_name])

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn(f"Element: {element_name}", output)
            self.assertIn("formats=[bgr, gray, rgb]", output)
            for expected_line in expected_lines:
                self.assertIn(expected_line, output)

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

    def test_cli_describe_displaysink_shows_autorange(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "displaysink"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: displaysink", output)
        self.assertIn("autorange: bool | optional | default=False", output)
        self.assertIn("depth: int | optional", output)

    def test_cli_describe_unknown_element_returns_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = cli_main(["describe", "missing"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Unknown element 'missing'", stderr.getvalue())

    def test_opencv_qt_environment_sets_safe_defaults(self) -> None:
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True):
            configure_opencv_qt_environment()

            self.assertEqual(os.environ["QT_QPA_PLATFORM"], "xcb")
            self.assertTrue(Path(os.environ["QT_QPA_FONTDIR"]).is_dir())

    def test_opencv_qt_environment_preserves_valid_user_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "XDG_SESSION_TYPE": "wayland",
                    "QT_QPA_PLATFORM": "wayland",
                    "QT_QPA_FONTDIR": tmp,
                },
                clear=True,
            ):
                configure_opencv_qt_environment()

                self.assertEqual(os.environ["QT_QPA_PLATFORM"], "wayland")
                self.assertEqual(os.environ["QT_QPA_FONTDIR"], tmp)

    def test_displaysink_uses_explicit_or_metadata_fps(self) -> None:
        frame = packet(np.zeros((4, 5, 3), dtype=np.uint8))
        metadata_fps_sink = DisplaySink("display", {"enabled": False})
        explicit_fps_sink = DisplaySink("display", {"enabled": False, "fps": 25})
        wait_sink = DisplaySink("display", {"enabled": False, "wait_ms": 7})

        self.assertEqual(metadata_fps_sink._wait_ms(frame), 33)
        self.assertEqual(explicit_fps_sink._wait_ms(frame), 40)
        self.assertEqual(wait_sink._wait_ms(frame), 7)

    def test_displaysink_autorange_false_returns_original_frame(self) -> None:
        sink = DisplaySink("display", {"enabled": False})
        frame = np.array([[100, 200]], dtype=np.uint16)

        self.assertIs(sink._display_frame(frame), frame)

    def test_displaysink_autorange_stretches_uint16_to_display_range(self) -> None:
        sink = DisplaySink("display", {"enabled": False, "autorange": True})
        frame = np.array([[100, 150, 200]], dtype=np.uint16)

        result = sink._display_frame(frame)

        self.assertEqual(result.dtype, np.uint16)
        self.assertEqual(result.tolist(), [[0, 32768, 65535]])
        self.assertEqual(frame.tolist(), [[100, 150, 200]])

    def test_displaysink_process_uses_autorange_frame_for_imshow(self) -> None:
        frame = packet(np.array([[100, 200]], dtype=np.uint16), fmt="gray")
        sink = DisplaySink("display", {"autorange": True, "wait_ms": 1})

        with (
            patch("src.sinks.displaysink.cv2.imshow") as imshow,
            patch("src.sinks.displaysink.cv2.waitKey", return_value=-1),
        ):
            sink.process({"in": frame})

        shown = imshow.call_args.args[1]
        self.assertEqual(shown.dtype, np.uint16)
        self.assertEqual(shown.tolist(), [[0, 65535]])
        self.assertEqual(frame.data.tolist(), [[100, 200]])

    def test_displaysink_depth_scales_active_bits_to_display_range(self) -> None:
        sink = DisplaySink("display", {"enabled": False, "depth": 14})
        frame = np.array([[0, 8192, 16383, 20000]], dtype=np.uint16)

        result = sink._display_frame(frame)

        self.assertEqual(result.dtype, np.uint16)
        self.assertEqual(result.tolist(), [[0, 32770, 65535, 65535]])
        self.assertEqual(frame.tolist(), [[0, 8192, 16383, 20000]])

    def test_displaysink_depth_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            DisplaySink("display", {"depth": 14, "autorange": True})
        with self.assertRaises(ValueError):
            DisplaySink("display", {"depth": 0})
        with self.assertRaises(ValueError):
            DisplaySink("display", {"depth": 9})._display_frame(
                np.zeros((1, 1), dtype=np.uint8)
            )

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
