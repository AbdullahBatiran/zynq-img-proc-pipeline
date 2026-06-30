from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from src.sinks.filesink import FileSink
from src.sources.filesrc import infer_frame_format, normalize_decoded_frame
from src.transformers.bit_shift import BitShift
from src.transformers.bilateral import Bilateral
from src.transformers.clahe import Clahe
from src.transformers.bypass import Bypass
from src.transformers.combine import Combine
from src.transformers.debug import Debug
from src.transformers.deflicker import Deflicker
from src.transformers.dog import Dog
from src.transformers.edge_enhance import EdgeEnhance
from src.transformers.dtype_convert import DtypeConvert
from src.transformers.fan_out import FanOut
from src.transformers.frame_diff_debug import FrameDiffDebug
from src.transformers.gaussian import Gaussian
from src.transformers.guided_filter import GuidedFilter
from src.transformers.hist_equalize import HistEqualize
from src.transformers.interlace_mimic_test import InterlaceMimicTest
from src.transformers.laplacian_sharp import LaplacianSharp
from src.transformers.linear_scale import LinearScale
from src.transformers.local_contrast import LocalContrast
from src.transformers.log_filter import LogFilter
from src.transformers.meam import Meam
from src.transformers.median import Median
from src.transformers.morphology import Morphology
from src.transformers.non_linear import NonLinear
from src.transformers.nl_means import NlMeans
from src.transformers.progress import Progress
from src.transformers.retinex import Retinex
from src.transformers.mono_to_color import MonoToColor
from src.transformers.resize import Resize
from src.transformers.rolling_background import RollingBackground
from src.transformers.sharpen_kernel import SharpenKernel
from src.transformers.temporal_denoise import TemporalDenoise
from src.transformers.tone_curve import ToneCurve
from src.transformers.text_overlay import TextOverlay
from src.transformers.unsharp import Unsharp
from src.transformers.tv_denoise import TvDenoise
from src.transformers.wavelet_denoise import WaveletDenoise


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


def _reference_meam(
    frame: np.ndarray,
    *,
    detail_gain: float,
    blur_sigma: float,
    output_bits: int = 16,
) -> np.ndarray:
    working = frame.astype(np.float32)
    base = cv2.GaussianBlur(working, (0, 0), sigmaX=blur_sigma)
    detail = working - base
    base_min = np.min(base)
    base_max = np.max(base)
    dynamic_range = base_max - base_min
    if dynamic_range == 0:
        dynamic_range = 1.0
    output_max = float((1 << output_bits) - 1)
    base_16bit = ((base - base_min) / dynamic_range) * output_max
    detail_16bit = (detail / dynamic_range) * output_max * detail_gain
    enhanced = base_16bit + detail_16bit
    return np.clip(np.rint(enhanced), 0, output_max).astype(np.uint16)


def _reference_non_linear(frame: np.ndarray, output_bits: int) -> np.ndarray:
    output_max = (1 << output_bits) - 1
    clipped = np.clip(frame, 0, output_max).astype(np.int64, copy=False)
    histogram = np.bincount(clipped.ravel(), minlength=output_max + 1).astype(
        np.uint64,
        copy=False,
    )
    modified = np.zeros_like(histogram, dtype=np.uint64)
    nonzero = histogram > 0
    modified[nonzero] = np.floor(
        np.log2(histogram[nonzero].astype(np.float64))
    ).astype(np.uint64)
    cumulative = np.cumsum(modified, dtype=np.uint64)
    total = int(cumulative[-1]) if cumulative.size else 0
    if total == 0:
        lut = np.zeros(output_max + 1, dtype=frame.dtype)
    else:
        lut = np.rint(cumulative.astype(np.float64) * output_max / total)
        lut = np.clip(lut, 0, output_max).astype(frame.dtype)
    return lut[clipped].astype(frame.dtype, copy=False)


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

    def test_non_linear_matches_reference_uint8_gray(self) -> None:
        frame = np.array(
            [
                [0, 0, 1, 2],
                [2, 2, 3, 3],
                [3, 4, 4, 4],
                [5, 6, 7, 7],
            ],
            dtype=np.uint8,
        )
        source = packet(frame, fmt="gray")
        result = NonLinear("nl", {"output-bits": 3}).process({"in": source})["out"][0]

        expected = _reference_non_linear(frame, output_bits=3)
        np.testing.assert_array_equal(result.data, expected)
        self.assertEqual(result.data.dtype, np.uint8)
        self.assertEqual(result.data.shape, frame.shape)
        self.assertEqual(result.metadata.depth, 8)
        self.assertEqual(result.metadata.format, "gray")
        self.assertEqual(result.metadata.channels, 1)
        self.assertEqual(result.metadata.extra["non_linear_by"], "nl")
        self.assertEqual(result.metadata.extra["non_linear_output_bits"], 3)
        self.assertEqual(result.metadata.extra["non_linear_output_max"], 7)
        self.assertEqual(result.metadata.extra["non_linear_levels"], 8)
        self.assertIn("non_linear_modified_total", result.metadata.extra)
        self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_non_linear_uint16_output_bits_14_keeps_container_depth(self) -> None:
        frame = np.array(
            [
                [0, 100, 200, 20000],
                [400, 400, 600, 30000],
                [800, 1000, 1200, 65535],
            ],
            dtype=np.uint16,
        )
        result = NonLinear("nl", {"output-bits": 14}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]

        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.data.shape, frame.shape)
        self.assertEqual(result.metadata.depth, 16)
        self.assertLessEqual(int(result.data.max()), 16383)
        self.assertEqual(result.metadata.extra["non_linear_output_bits"], 14)
        self.assertEqual(result.metadata.extra["non_linear_output_max"], 16383)
        self.assertEqual(result.metadata.extra["non_linear_levels"], 16384)

    def test_non_linear_zero_modified_histogram_outputs_zero(self) -> None:
        frame = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        result = NonLinear("nl", {"output-bits": 4}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]

        self.assertEqual(result.data.tolist(), np.zeros_like(frame).tolist())
        self.assertEqual(result.metadata.extra["non_linear_modified_total"], 0)

    def test_non_linear_accepts_hxwx1_and_preserves_shape(self) -> None:
        frame = np.array([[[0], [1]], [[1], [2]]], dtype=np.uint8)
        result = NonLinear("nl", {"output-bits": 2}).process(
            {"in": packet(frame, fmt="gray")}
        )["out"][0]

        self.assertEqual(result.data.shape, frame.shape)
        self.assertEqual(result.metadata.channels, 1)

    def test_non_linear_rejects_invalid_parameters_and_frames(self) -> None:
        for params in ({"output-bits": 0}, {"output-bits": 17}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                NonLinear("nl", params)

        transform = NonLinear("nl", {})
        invalid_frames = [
            packet(np.zeros((2, 2), dtype=np.float32), fmt="gray"),
            packet(np.zeros((2, 2), dtype=np.uint8), fmt="bgr"),
            packet(np.zeros((2, 2, 3), dtype=np.uint8), fmt="gray"),
        ]
        for invalid in invalid_frames:
            with self.subTest(metadata=invalid.metadata), self.assertRaises(ValueError):
                transform.process({"in": invalid})
        with self.assertRaises(ValueError):
            NonLinear("nl", {"output-bits": 9}).process(
                {"in": packet(np.zeros((2, 2), dtype=np.uint8), fmt="gray")}
            )

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

    def test_meam_matches_reference_math_and_outputs_uint16(self) -> None:
        frame = np.array(
            [
                [5000, 5010, 5020, 5030],
                [5040, 5400, 5410, 5060],
                [5070, 5420, 5430, 5090],
                [5100, 5110, 5120, 5130],
            ],
            dtype=np.uint16,
        )
        source = packet(frame, fmt="gray")
        transform = Meam("meam", {"detail-gain": 4.0, "blur-sigma": 1.5})

        result = transform.process({"in": source})["out"][0]

        expected = _reference_meam(frame, detail_gain=4.0, blur_sigma=1.5)
        np.testing.assert_array_equal(result.data, expected)
        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.metadata.depth, 16)
        self.assertEqual(result.metadata.format, "gray")
        self.assertEqual(result.metadata.channels, 1)
        self.assertEqual(result.metadata.extra["enhanced_by"], "meam")
        self.assertEqual(result.metadata.extra["enhancement_name"], "meam")
        self.assertEqual(
            result.metadata.extra["enhancement_params"],
            {"detail-gain": 4.0, "blur-sigma": 1.5, "output-bits": 16},
        )
        self.assertIn("meam_dynamic_range", result.metadata.extra)
        self.assertEqual(result.metadata.extra["meam_output_bits"], 16)
        self.assertEqual(result.metadata.extra["meam_output_max"], 65535)
        self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_meam_output_bits_limits_effective_range(self) -> None:
        frame = np.array(
            [
                [5000, 5010, 5020, 5030],
                [5040, 5400, 5410, 5060],
                [5070, 5420, 5430, 5090],
                [5100, 5110, 5120, 5130],
            ],
            dtype=np.uint16,
        )
        result = Meam(
            "meam", {"detail-gain": 4.0, "blur-sigma": 1.5, "output-bits": 14}
        ).process({"in": packet(frame, fmt="gray")})["out"][0]

        expected = _reference_meam(
            frame, detail_gain=4.0, blur_sigma=1.5, output_bits=14
        )
        np.testing.assert_array_equal(result.data, expected)
        self.assertEqual(result.data.dtype, np.uint16)
        self.assertLessEqual(int(result.data.max()), 16383)
        self.assertEqual(result.metadata.depth, 16)
        self.assertEqual(result.metadata.extra["meam_output_bits"], 14)
        self.assertEqual(result.metadata.extra["meam_output_max"], 16383)

    def test_meam_flat_frame_outputs_zero_and_accepts_aliases(self) -> None:
        frame = np.full((3, 4, 1), 5000, dtype=np.uint16)
        result = Meam(
            "meam",
            {"detail_gain": 2.0, "blur_sigma": 3.0, "output_bits": 12},
        ).process({"in": packet(frame, fmt="gray")})["out"][0]

        self.assertEqual(result.data.shape, (3, 4))
        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(int(result.data.max()), 0)
        self.assertEqual(result.metadata.depth, 16)
        self.assertEqual(result.metadata.extra["meam_dynamic_range"], 1.0)
        self.assertEqual(result.metadata.extra["meam_output_bits"], 12)
        self.assertEqual(result.metadata.extra["meam_output_max"], 4095)

    def test_meam_rejects_invalid_parameters_and_frames(self) -> None:
        for params in (
            {"detail-gain": -0.1},
            {"blur-sigma": 0},
            {"output-bits": 0},
            {"output-bits": 17},
            {"output_bits": 14, "output-bits": 12},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                Meam("meam", params)

        transform = Meam("meam", {})
        invalid_frames = [
            packet(np.zeros((2, 2), dtype=np.uint8), fmt="gray"),
            packet(np.zeros((2, 2), dtype=np.uint16), fmt="bgr"),
            packet(np.zeros((2, 2, 3), dtype=np.uint16), fmt="gray"),
        ]
        for invalid in invalid_frames:
            with self.subTest(metadata=invalid.metadata), self.assertRaises(ValueError):
                transform.process({"in": invalid})

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

    def test_sharpen_kernel_cross_and_full_match_filter2d(self) -> None:
        frame = np.array(
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
            ],
            dtype=np.uint8,
        )
        kernels = {
            "cross": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64),
            "full": np.array(
                [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
                dtype=np.float64,
            ),
        }

        for kernel_name, kernel in kernels.items():
            with self.subTest(kernel=kernel_name):
                source = packet(frame, fmt="gray")
                result = SharpenKernel("sharp", {"kernel": kernel_name}).process(
                    {"in": source}
                )["out"][0]
                expected = cv2.filter2D(
                    frame.astype(np.float64),
                    cv2.CV_64F,
                    kernel,
                )
                expected = np.clip(expected, 0, 255).astype(np.uint8)

                np.testing.assert_array_equal(result.data, expected)
                self.assertEqual(result.data.dtype, np.uint8)
                self.assertEqual(result.metadata.depth, 8)
                self.assertEqual(result.metadata.format, "gray")
                self.assertEqual(result.metadata.extra["filtered_by"], "sharp")
                self.assertEqual(result.metadata.extra["filter_name"], "sharpen-kernel")
                self.assertEqual(
                    result.metadata.extra["filter_params"],
                    {
                        "kernel": kernel_name,
                        "iterations": 1,
                        "range-mode": "clip",
                        "output-bits": 8,
                    },
                )
                self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_sharpen_kernel_supports_uint16_color_and_iterations(self) -> None:
        frame = (np.arange(3 * 4 * 3, dtype=np.uint16).reshape((3, 4, 3)) * 100)
        result = SharpenKernel("sharp", {"iterations": 2}).process(
            {"in": packet(frame, fmt="rgb")}
        )["out"][0]

        self.assertEqual(result.data.shape, frame.shape)
        self.assertEqual(result.data.dtype, np.uint16)
        self.assertEqual(result.metadata.depth, 16)
        self.assertEqual(result.metadata.channels, 3)
        self.assertEqual(result.metadata.format, "rgb")
        self.assertEqual(result.metadata.extra["filter_params"]["iterations"], 2)
        self.assertEqual(result.metadata.extra["filter_params"]["range-mode"], "clip")
        self.assertEqual(result.metadata.extra["filter_params"]["output-bits"], 16)

    def test_sharpening_limit_mode_respects_output_bits_range(self) -> None:
        frame = np.array(
            [
                [0, 1000, 16000],
                [2000, 12000, 20000],
                [4000, 14000, 24000],
            ],
            dtype=np.uint16,
        )
        transforms = [
            Unsharp(
                "sharp",
                {
                    "amount": 2.0,
                    "kernel-size": 3,
                    "range-mode": "limit",
                    "output-bits": 14,
                },
            ),
            LaplacianSharp(
                "sharp",
                {"amount": 1.0, "range-mode": "limit", "output-bits": 14},
            ),
            SharpenKernel(
                "sharp",
                {"kernel": "full", "range-mode": "limit", "output-bits": 14},
            ),
        ]

        for transform in transforms:
            with self.subTest(transform=transform.type_name):
                source = packet(frame, fmt="gray")
                result = transform.process({"in": source})["out"][0]

                self.assertEqual(result.data.dtype, np.uint16)
                self.assertEqual(result.data.shape, frame.shape)
                self.assertEqual(result.metadata.depth, 16)
                self.assertEqual(result.metadata.format, "gray")
                self.assertEqual(result.metadata.channels, 1)
                self.assertLessEqual(int(result.data.max()), 16383)
                self.assertGreaterEqual(int(result.data.min()), 0)
                self.assertEqual(
                    result.metadata.extra["filter_params"]["range-mode"], "limit"
                )
                self.assertEqual(
                    result.metadata.extra["filter_params"]["output-bits"], 14
                )
                self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_sharpen_kernel_rejects_invalid_parameters(self) -> None:
        for params in (
            {"kernel": "laplacian"},
            {"iterations": 0},
            {"range-mode": "bad"},
            {"output-bits": 0},
            {"output-bits": 17},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                SharpenKernel("sharp", params)

    def test_filters_preserve_three_channel_shape(self) -> None:
        frame = (np.arange(5 * 5 * 3, dtype=np.uint16).reshape((5, 5, 3)) * 100)
        cases = [
            Unsharp("f", {"kernel-size": 3}),
            Median("f", {"kernel-size": 3}),
            Gaussian("f", {"kernel-size": 3}),
            Bilateral("f", {"diameter": 3, "sigma-color": 100.0}),
            LaplacianSharp("f", {"kernel-size": 3}),
            SharpenKernel("f", {}),
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
            (Unsharp, {"range-mode": "bad"}),
            (Unsharp, {"output-bits": 0}),
            (Unsharp, {"output-bits": 17}),
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
            (LaplacianSharp, {"range-mode": "bad"}),
            (LaplacianSharp, {"output-bits": 0}),
            (LaplacianSharp, {"output-bits": 17}),
            (SharpenKernel, {"kernel": "unknown"}),
            (SharpenKernel, {"iterations": 0}),
            (SharpenKernel, {"range-mode": "bad"}),
            (SharpenKernel, {"output-bits": 0}),
            (SharpenKernel, {"output-bits": 17}),
        ]
        for transform_cls, params in invalid_configurations:
            with self.subTest(transform=transform_cls.__name__, params=params):
                with self.assertRaises(ValueError):
                    transform_cls("f", params)

        with self.assertRaises(ValueError):
            Median("f", {"kernel-size": 7}).process(
                {"in": packet(np.zeros((5, 5), dtype=np.uint16), fmt="gray")}
            )
        with self.assertRaises(ValueError):
            Unsharp("f", {"output-bits": 9}).process(
                {"in": packet(np.zeros((5, 5), dtype=np.uint8), fmt="gray")}
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

    def test_ir_enhancement_elements_preserve_dtype_shape_and_depth(self) -> None:
        transforms = [
            Clahe("e", {"tile-grid-size": 4}),
            ToneCurve("e", {"mode": "gamma", "gamma": 0.8}),
            Retinex("e", {"mode": "single", "sigma": 3.0, "output-mode": "normalize"}),
            LocalContrast("e", {"sigma": 3.0, "normalize": True}),
            RollingBackground("e", {"radius": 3, "normalize": True}),
            Morphology("e", {"op": "tophat", "kernel-size": 3}),
            Dog("e", {"sigma-small": 1.0, "sigma-large": 2.0}),
            LogFilter("e", {"sigma": 1.0, "kernel-size": 3}),
            EdgeEnhance("e", {"operator": "sobel", "ksize": 3}),
            GuidedFilter("e", {"radius": 2, "mode": "enhance"}),
            NlMeans("e", {"h": 5.0, "template-window-size": 3, "search-window-size": 5}),
            TvDenoise("e", {"weight": 0.05, "max-num-iter": 5}),
            WaveletDenoise("e", {"sigma": 0.01}),
            TemporalDenoise("e", {"mode": "mean", "window": 3}),
            Deflicker("e", {"mode": "percentile", "alpha": 0.5}),
        ]
        frames = [
            np.tile(np.arange(16, dtype=np.uint8), (16, 1)),
            np.tile(np.arange(16, dtype=np.uint16), (16, 1)) * 100,
        ]

        for transform in transforms:
            for frame in frames:
                with self.subTest(element=transform.type_name, dtype=frame.dtype):
                    source = packet(frame, fmt="gray")
                    result = transform.process({"in": source})["out"][0]

                    self.assertEqual(result.data.dtype, frame.dtype)
                    self.assertEqual(result.data.shape, frame.shape)
                    self.assertEqual(result.metadata.depth, source.metadata.depth)
                    self.assertEqual(result.metadata.extra["enhanced_by"], "e")
                    self.assertEqual(
                        result.metadata.extra["enhancement_name"],
                        transform.type_name,
                    )
                    self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_ir_enhancement_elements_preserve_three_channel_shape(self) -> None:
        frame = np.dstack(
            [
                np.tile(np.arange(16, dtype=np.uint8), (16, 1)),
                np.tile(np.arange(16, dtype=np.uint8), (16, 1)).T,
                np.full((16, 16), 32, dtype=np.uint8),
            ]
        )
        transforms = [
            Clahe("e", {"tile-grid-size": 4}),
            ToneCurve("e", {"mode": "log"}),
            LocalContrast("e", {"sigma": 3.0}),
            Morphology("e", {"op": "gradient", "kernel-size": 3}),
            Dog("e", {"sigma-large": 2.0}),
            EdgeEnhance("e", {}),
            GuidedFilter("e", {"radius": 2}),
            TvDenoise("e", {"max-num-iter": 5}),
            WaveletDenoise("e", {"sigma": 0.01}),
        ]

        for transform in transforms:
            with self.subTest(element=transform.type_name):
                result = transform.process({"in": packet(frame, fmt="rgb")})["out"][0]
                self.assertEqual(result.data.dtype, np.uint8)
                self.assertEqual(result.data.shape, frame.shape)
                self.assertEqual(result.metadata.channels, 3)

    def test_ir_enhancement_algorithm_behaviors(self) -> None:
        gradient = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
        clahe = Clahe("e", {"tile-grid-size": 4}).process(
            {"in": packet(gradient, fmt="gray")}
        )["out"][0]
        self.assertFalse(np.array_equal(clahe.data, gradient))

        tone = ToneCurve("e", {"mode": "gamma", "gamma": 2.0}).process(
            {"in": packet(np.array([[0, 128, 255]], dtype=np.uint8), fmt="gray")}
        )["out"][0]
        self.assertLess(int(tone.data[0, 1]), 128)

        target = np.zeros((9, 9), dtype=np.uint8)
        target[4, 4] = 255
        top_hat = Morphology("e", {"op": "tophat", "kernel-size": 3}).process(
            {"in": packet(target, fmt="gray")}
        )["out"][0]
        self.assertEqual(int(top_hat.data[4, 4]), 255)

        edge_frame = np.zeros((16, 16), dtype=np.uint8)
        edge_frame[:, 8:] = 200
        for transform in (
            Dog("e", {"sigma-small": 1.0, "sigma-large": 2.0}),
            LogFilter("e", {"kernel-size": 3}),
            EdgeEnhance("e", {}),
        ):
            with self.subTest(element=transform.type_name):
                result = transform.process({"in": packet(edge_frame, fmt="gray")})[
                    "out"
                ][0]
                self.assertGreater(int(result.data.max()), 0)

    def test_temporal_enhancement_is_deterministic(self) -> None:
        first = packet(np.full((2, 2), 10, dtype=np.uint8), fmt="gray", index=0)
        second = packet(np.full((2, 2), 20, dtype=np.uint8), fmt="gray", index=1)
        transform = TemporalDenoise("t", {"mode": "mean", "window": 2})

        out1 = transform.process({"in": first})["out"][0]
        out2 = transform.process({"in": second})["out"][0]

        self.assertEqual(int(out1.data[0, 0]), 10)
        self.assertEqual(int(out2.data[0, 0]), 15)
        self.assertEqual(out2.metadata.extra["temporal_history_size"], 2)

    def test_debug_progress_and_frame_diff_diagnostics(self) -> None:
        frame0 = packet(np.zeros((2, 2), dtype=np.uint8), fmt="gray", index=0)
        frame1 = packet(np.ones((2, 2), dtype=np.uint8) * 4, fmt="gray", index=1)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = Debug(
                "dbg",
                {
                    "show-percentiles": True,
                    "percentiles": "0,0.5,1",
                    "show-histogram": True,
                    "hist-bins": 2,
                },
            ).process({"in": frame1})["out"][0]
        self.assertIs(result, frame1)
        self.assertIn("percentiles", stdout.getvalue())
        self.assertIn("histogram bins=2", stdout.getvalue())

        progress = Progress("p", {"every-frames": 1})
        progress.start(PipelineContext())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertIs(progress.process({"in": frame0})["out"][0], frame0)
        self.assertIn("[progress p]", stdout.getvalue())

        diff = FrameDiffDebug("d", {"every-frames": 1})
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            diff.process({"in": frame0})
            self.assertIs(diff.process({"in": frame1})["out"][0], frame1)
        self.assertIn("mean_abs=4", stdout.getvalue())

    def test_new_ir_elements_reject_invalid_parameters(self) -> None:
        invalid = [
            (Clahe, {"clip-limit": 0}),
            (ToneCurve, {"mode": "bad"}),
            (Retinex, {"sigma": 0}),
            (LocalContrast, {"mode": "bad"}),
            (RollingBackground, {"radius": 0}),
            (Morphology, {"op": "bad"}),
            (Dog, {"sigma-small": 3, "sigma-large": 1}),
            (LogFilter, {"kernel-size": 2}),
            (EdgeEnhance, {"operator": "bad"}),
            (GuidedFilter, {"radius": 0}),
            (NlMeans, {"h": 0}),
            (TvDenoise, {"weight": 0}),
            (WaveletDenoise, {"wavelet": "db2"}),
            (TemporalDenoise, {"window": 0}),
            (Deflicker, {"low-perc": 0.9, "high-perc": 0.1}),
            (Progress, {"every-frames": 0}),
            (FrameDiffDebug, {"every-frames": 0}),
        ]
        for transform_cls, params in invalid:
            with self.subTest(transform=transform_cls.__name__, params=params):
                with self.assertRaises(ValueError):
                    transform_cls("bad", params)

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
            result = DtypeConvert(
                "dtype", {"dtype": "uint16", "overflow": "clamp"}
            ).process(
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
        self.assertEqual(result.metadata.extra["dtype_convert_overflow"], "clamp")
        self.assertTrue(result.metadata.extra["dtype_convert_warn"])
        self.assertIn(source.metadata.packet_id, result.metadata.parents)

    def test_dtype_convert_uint16_to_uint8_clips_and_warns(self) -> None:
        frame = np.array([[0, 255, 256, 1000]], dtype=np.uint16)
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            result = DtypeConvert("dtype", {"dtype": "uint8", "overflow": "clamp"}).process(
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
        self.assertEqual(result.metadata.extra["dtype_convert_overflow"], "clamp")
        self.assertTrue(result.metadata.extra["dtype_convert_warn"])

    def test_dtype_convert_clamp_can_suppress_warning(self) -> None:
        frame = np.array([[0, 255, 256, 1000]], dtype=np.uint16)
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            result = DtypeConvert(
                "dtype", {"dtype": "uint8", "overflow": "clamp", "warn": False}
            ).process({"in": packet(frame, fmt="gray")})["out"][0]

        self.assertEqual(result.data.dtype, np.uint8)
        self.assertEqual(result.data.tolist(), [[0, 255, 255, 255]])
        self.assertEqual(stderr.getvalue(), "")
        self.assertFalse(result.metadata.extra["dtype_convert_warn"])

    def test_dtype_convert_uint32_color_to_uint16_clips(self) -> None:
        frame = np.array(
            [[[0, 1, 65535], [65536, 70000, 100000]]],
            dtype=np.uint32,
        )
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            result = DtypeConvert(
                "dtype", {"dtype": "uint16", "overflow": "clamp"}
            ).process(
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

    def test_dtype_convert_wrap_matches_numpy_cast_without_warning(self) -> None:
        frame = np.array([[0, 255, 256, 1000]], dtype=np.uint16)
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            result = DtypeConvert("dtype", {"dtype": "uint8", "overflow": "wrap"}).process(
                {"in": packet(frame, fmt="gray")}
            )["out"][0]

        np.testing.assert_array_equal(result.data, frame.astype(np.uint8))
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(result.metadata.extra["dtype_convert_overflow"], "wrap")
        self.assertTrue(result.metadata.extra["dtype_convert_warn"])

    def test_dtype_convert_rejects_invalid_parameters_and_frames(self) -> None:
        with self.assertRaises(ValueError):
            DtypeConvert("dtype", {"dtype": "float32"})
        with self.assertRaises(ValueError):
            DtypeConvert("dtype", {"dtype": "uint8"})
        with self.assertRaises(ValueError):
            DtypeConvert("dtype", {"dtype": "uint8", "overflow": "bad"})

        transform = DtypeConvert("dtype", {"dtype": "uint8", "overflow": "clamp"})
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

    def test_interlace_mimic_test_emits_one_frame_per_two_inputs(self) -> None:
        transform = InterlaceMimicTest("interlace", {})
        first = packet(np.full((4, 3), 1, dtype=np.uint8), fmt="gray", index=10)
        second = packet(np.full((4, 3), 2, dtype=np.uint8), fmt="gray", index=11)

        first_result = transform.process({"in": first})
        second_result = transform.process({"in": second})

        self.assertEqual(first_result, {})
        result = second_result["out"][0]
        self.assertEqual(result.data.tolist(), [
            [1, 1, 1],
            [2, 2, 2],
            [1, 1, 1],
            [2, 2, 2],
        ])
        self.assertIsNot(result, first)
        self.assertEqual(result.metadata.index, 0)
        self.assertEqual(result.metadata.pts, first.metadata.pts)
        self.assertEqual(result.metadata.fps, 15.0)
        self.assertIn(first.metadata.packet_id, result.metadata.parents)
        self.assertIn(second.metadata.packet_id, result.metadata.parents)
        self.assertEqual(
            result.metadata.extra["interlace_mimic_test_by"],
            "interlace",
        )
        self.assertEqual(result.metadata.extra["interlace_mimic_pair_index"], 0)
        self.assertEqual(result.metadata.extra["interlace_mimic_first_index"], 10)
        self.assertEqual(result.metadata.extra["interlace_mimic_second_index"], 11)
        self.assertEqual(
            result.metadata.extra["interlace_mimic_line_order"],
            "first_rows_0_even_second_rows_1_odd",
        )

    def test_interlace_mimic_test_odd_height_and_next_pair(self) -> None:
        transform = InterlaceMimicTest("interlace", {})
        first = packet(np.full((5, 2), 10, dtype=np.uint8), fmt="gray")
        second = packet(np.full((5, 2), 20, dtype=np.uint8), fmt="gray")
        third = packet(np.full((5, 2), 30, dtype=np.uint8), fmt="gray")
        fourth = packet(np.full((5, 2), 40, dtype=np.uint8), fmt="gray")

        first_pair = transform.process({"in": first})
        first_pair = transform.process({"in": second})["out"][0]
        third_result = transform.process({"in": third})
        second_pair = transform.process({"in": fourth})["out"][0]

        self.assertEqual(first_pair.data.tolist(), [
            [10, 10],
            [20, 20],
            [10, 10],
            [20, 20],
            [10, 10],
        ])
        self.assertEqual(third_result, {})
        self.assertEqual(second_pair.data.tolist(), [
            [30, 30],
            [40, 40],
            [30, 30],
            [40, 40],
            [30, 30],
        ])
        self.assertEqual(second_pair.metadata.index, 1)
        self.assertEqual(
            second_pair.metadata.extra["interlace_mimic_pair_index"],
            1,
        )

    def test_interlace_mimic_test_stop_drops_unmatched_frame(self) -> None:
        transform = InterlaceMimicTest("interlace", {})
        orphan = packet(np.full((2, 2), 1, dtype=np.uint8), fmt="gray")
        first = packet(np.full((2, 2), 2, dtype=np.uint8), fmt="gray")
        second = packet(np.full((2, 2), 3, dtype=np.uint8), fmt="gray")

        self.assertEqual(transform.process({"in": orphan}), {})
        transform.stop()
        self.assertEqual(transform.process({"in": first}), {})
        result = transform.process({"in": second})["out"][0]

        self.assertEqual(result.data.tolist(), [[2, 2], [3, 3]])

    def test_interlace_mimic_test_rejects_mismatched_pairs(self) -> None:
        cases = [
            (
                packet(np.zeros((2, 2), dtype=np.uint8), fmt="gray"),
                packet(np.zeros((3, 2), dtype=np.uint8), fmt="gray"),
            ),
            (
                packet(np.zeros((2, 2), dtype=np.uint8), fmt="gray"),
                packet(np.zeros((2, 2), dtype=np.uint16), fmt="gray"),
            ),
            (
                packet(np.zeros((2, 2), dtype=np.uint8), fmt="gray"),
                packet(np.zeros((2, 2), dtype=np.uint8), fmt="bgr"),
            ),
        ]
        base = packet(np.zeros((2, 2), dtype=np.uint8), fmt="gray")
        depth_mismatch = FramePacket(
            data=base.data,
            metadata=base.metadata.derive(depth=16),
        )
        channel_mismatch = FramePacket(
            data=base.data,
            metadata=base.metadata.derive(channels=2),
        )
        cases.extend([(base, depth_mismatch), (base, channel_mismatch)])

        for first, second in cases:
            with self.subTest(second=second.metadata), self.assertRaises(ValueError):
                transform = InterlaceMimicTest("interlace", {})
                transform.process({"in": first})
                transform.process({"in": second})

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
            "filesrc path=in.mp4 ! Unsharp kernel-size=3 range-mode=limit "
            "output-bits=8 ! gaussian sigma-x=1.0 "
            "! bilateral sigma-color=25 sigma-space=3 ! laplacian-sharp "
            "kernel-size=3 iterations=2 ! sharpen-kernel kernel=full "
            "iterations=2 range-mode=limit output-bits=8 ! filesink path=out.mp4"
        )

        self.assertEqual(spec.elements[1].type, "Unsharp")
        self.assertEqual(spec.elements[1].params["kernel-size"], 3)
        self.assertEqual(spec.elements[1].params["range-mode"], "limit")
        self.assertEqual(spec.elements[1].params["output-bits"], 8)
        self.assertEqual(spec.elements[2].type, "gaussian")
        self.assertEqual(spec.elements[2].params["sigma-x"], 1.0)
        self.assertEqual(spec.elements[3].type, "bilateral")
        self.assertEqual(spec.elements[3].params["sigma-color"], 25)
        self.assertEqual(spec.elements[3].params["sigma-space"], 3)
        self.assertEqual(spec.elements[4].type, "laplacian-sharp")
        self.assertEqual(spec.elements[4].params["kernel-size"], 3)
        self.assertEqual(spec.elements[4].params["iterations"], 2)
        self.assertEqual(spec.elements[5].type, "sharpen-kernel")
        self.assertEqual(spec.elements[5].params["kernel"], "full")
        self.assertEqual(spec.elements[5].params["iterations"], 2)
        self.assertEqual(spec.elements[5].params["range-mode"], "limit")
        self.assertEqual(spec.elements[5].params["output-bits"], 8)

    def test_cli_parser_accepts_hyphenated_ir_enhancement_params(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mp4 ! clahe clip-limit=2.0 tile-grid-size=4 "
            "! meam detail-gain=4.0 blur-sigma=10.0 output-bits=14 "
            "! non-linear output-bits=14 ! dog sigma-small=1.0 "
            "sigma-large=3.0 ! progress every-frames=30 ! filesink path=out.mp4"
        )

        self.assertEqual(spec.elements[1].type, "clahe")
        self.assertEqual(spec.elements[1].params["clip-limit"], 2.0)
        self.assertEqual(spec.elements[1].params["tile-grid-size"], 4)
        self.assertEqual(spec.elements[2].type, "meam")
        self.assertEqual(spec.elements[2].params["detail-gain"], 4.0)
        self.assertEqual(spec.elements[2].params["blur-sigma"], 10.0)
        self.assertEqual(spec.elements[2].params["output-bits"], 14)
        self.assertEqual(spec.elements[3].type, "non-linear")
        self.assertEqual(spec.elements[3].params["output-bits"], 14)
        self.assertEqual(spec.elements[4].type, "dog")
        self.assertEqual(spec.elements[4].params["sigma-small"], 1.0)
        self.assertEqual(spec.elements[4].params["sigma-large"], 3.0)
        self.assertEqual(spec.elements[5].type, "progress")
        self.assertEqual(spec.elements[5].params["every-frames"], 30)

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
            "filesrc path=in.mkv ! dtype-convert dtype=uint16 overflow=clamp warn=false "
            "! filesink path=out.mp4"
        )

        self.assertEqual(spec.elements[1].type, "dtype-convert")
        self.assertEqual(spec.elements[1].params["dtype"], "uint16")
        self.assertEqual(spec.elements[1].params["overflow"], "clamp")
        self.assertFalse(spec.elements[1].params["warn"])

    def test_cli_parser_accepts_filesink_quality(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mkv ! filesink path=out.avi codec=MJPG quality=95"
        )

        self.assertEqual(spec.elements[1].type, "filesink")
        self.assertEqual(spec.elements[1].params["codec"], "MJPG")
        self.assertEqual(spec.elements[1].params["quality"], 95)

    def test_cli_parser_accepts_bypass(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mkv ! bypass ! filesink path=out.mp4"
        )

        self.assertEqual(spec.elements[1].type, "bypass")
        self.assertEqual(spec.elements[1].params, {})

    def test_cli_parser_accepts_interlace_mimic_test(self) -> None:
        spec = parse_pipeline_expression(
            "filesrc path=in.mkv ! interlace-mimic-test ! filesink path=out.mp4"
        )

        self.assertEqual(spec.elements[1].type, "interlace-mimic-test")
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
        self.assertIn("combine", output)
        self.assertIn("Compose", output)
        self.assertIn("fan-out", output)
        self.assertIn("temporal-denoise", output)
        self.assertIn("Control", output)
        self.assertIn("hist_equalize", output)
        self.assertIn("Contrast", output)
        self.assertIn("linear-scale", output)
        self.assertIn("clahe", output)
        self.assertIn("meam", output)
        self.assertIn("non-linear", output)
        self.assertIn("tone-curve", output)
        self.assertIn("debug", output)
        self.assertIn("progress", output)
        self.assertIn("Debug", output)
        self.assertIn("bilateral", output)
        self.assertIn("Filter", output)
        self.assertIn("morphology", output)
        self.assertIn("sharpen-kernel", output)
        self.assertIn("wavelet-denoise", output)
        self.assertIn("resize", output)
        self.assertIn("Geometry", output)
        self.assertIn("combine               Compose", output)
        self.assertIn("text-overlay          Compose", output)
        self.assertIn("mono-to-color         Color", output)
        self.assertIn("bypass                Control", output)
        self.assertIn("fan-out               Control", output)
        self.assertIn("hist_equalize         Contrast", output)
        self.assertIn("linear-scale          Contrast", output)
        self.assertIn("non-linear            Contrast", output)
        self.assertIn("debug                 Debug", output)
        self.assertIn("bilateral             Filter", output)
        self.assertIn("resize                Geometry", output)
        self.assertIn("bit-shift             Intensity", output)
        self.assertIn("dtype-convert         Intensity", output)
        self.assertIn("interlace-mimic-test  Test", output)
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
        self.assertIn("    fan-out", output)
        self.assertIn("    temporal-denoise", output)
        self.assertIn("    deflicker", output)
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
        self.assertIn("  Contrast\n    clahe", output)
        self.assertIn("    hist_equalize", output)
        self.assertIn("    meam", output)
        self.assertIn("    non-linear", output)
        self.assertIn("Parameters: none", output)
        self.assertIn("    linear-scale", output)
        self.assertIn("    local-contrast", output)
        self.assertIn("    retinex", output)
        self.assertIn("    rolling-background", output)
        self.assertIn("    tone-curve", output)
        self.assertIn("  Debug\n    debug", output)
        self.assertIn("    progress", output)
        self.assertIn("    frame-diff-debug", output)
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
        self.assertIn("    morphology", output)
        self.assertIn("    nl-means", output)
        self.assertIn("    sharpen-kernel", output)
        self.assertIn("    tv-denoise", output)
        self.assertIn("    wavelet-denoise", output)
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
        self.assertIn("  Test\n    interlace-mimic-test", output)
        self.assertIn("Sinks\n  File\n    filesink", output)
        self.assertIn("  GUI\n    displaysink", output)

    def test_cli_describe_filesink_shows_quality_option(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "filesink"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: filesink", output)
        self.assertIn("Subcategory: File", output)
        self.assertIn("path: path | required", output)
        self.assertIn("codec: str | optional", output)
        self.assertIn("quality: int | optional", output)

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

    def test_cli_describe_meam_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "meam"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: meam", output)
        self.assertIn("Subcategory: Contrast", output)
        self.assertIn("detail-gain: float | optional", output)
        self.assertIn("blur-sigma: float | optional", output)
        self.assertIn("output-bits: int | optional", output)
        self.assertIn("formats=[gray]", output)
        self.assertIn("depths=[16]", output)
        self.assertEqual(output.count("depths=[16]"), 2)

    def test_cli_describe_non_linear_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "non-linear"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: non-linear", output)
        self.assertIn("Subcategory: Contrast", output)
        self.assertIn("output-bits: int | optional", output)
        self.assertIn("formats=[gray]", output)
        self.assertEqual(output.count("depths=[8, 16]"), 2)

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
        self.assertIn("overflow: str | required", output)
        self.assertIn("warn: bool | optional", output)
        self.assertIn("choices=[uint8, uint16, uint32]", output)
        self.assertIn("choices=[clamp, wrap]", output)
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

    def test_cli_describe_interlace_mimic_test_shows_element_details(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["describe", "interlace-mimic-test"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Element: interlace-mimic-test", output)
        self.assertIn("Subcategory: Test", output)
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
            "unsharp": [
                "amount: float | optional",
                "kernel-size: int | optional",
                "range-mode: str | optional",
                "output-bits: int | optional",
            ],
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
                "range-mode: str | optional",
                "output-bits: int | optional",
            ],
            "sharpen-kernel": [
                "kernel: str | optional",
                "iterations: int | optional",
                "range-mode: str | optional",
                "output-bits: int | optional",
            ],
            "clahe": ["clip-limit: float | optional", "tile-grid-size: int | optional"],
            "tone-curve": ["mode: str | optional", "input-max: number | optional"],
            "retinex": ["sigmas: list | optional", "output-mode: str | optional"],
            "local-contrast": ["epsilon: float | optional", "normalize: bool | optional"],
            "rolling-background": ["radius: int | optional"],
            "morphology": ["op: str | optional", "kernel-shape: str | optional"],
            "dog": ["sigma-small: float | optional", "sigma-large: float | optional"],
            "log-filter": ["kernel-size: int | optional", "normalize: bool | optional"],
            "edge-enhance": ["operator: str | optional", "ksize: int | optional"],
            "guided-filter": ["radius: int | optional", "eps: float | optional"],
            "nl-means": [
                "template-window-size: int | optional",
                "search-window-size: int | optional",
            ],
            "tv-denoise": ["max-num-iter: int | optional"],
            "wavelet-denoise": [
                "wavelet: str | optional",
                "rescale-sigma: bool | optional",
            ],
            "temporal-denoise": ["window: int | optional", "alpha: float | optional"],
            "deflicker": ["low-perc: float | optional", "high-perc: float | optional"],
        }
        unconstrained_elements = {
            "progress": ["every-frames: int | optional", "every-seconds: float | optional"],
            "frame-diff-debug": ["show-mean-abs: bool | optional"],
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

        for element_name, expected_lines in unconstrained_elements.items():
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(["describe", element_name])

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn(f"Element: {element_name}", output)
            self.assertIn("in: FramePacket", output)
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

    def test_filesink_sets_quality_when_requested(self) -> None:
        frame = packet(np.zeros((4, 5, 3), dtype=np.uint8))
        writer = MagicMock()
        writer.isOpened.return_value = True
        writer.set.return_value = True

        with patch("src.sinks.filesink.cv2.VideoWriter", return_value=writer):
            FileSink(
                "out",
                {"path": "out.avi", "codec": "MJPG", "quality": 95},
            ).process({"in": frame})

        writer.set.assert_called_once_with(cv2.VIDEOWRITER_PROP_QUALITY, 95.0)
        writer.write.assert_called_once_with(frame.data)

    def test_filesink_warns_when_quality_is_not_supported(self) -> None:
        frame = packet(np.zeros((4, 5, 3), dtype=np.uint8))
        writer = MagicMock()
        writer.isOpened.return_value = True
        writer.set.return_value = False
        stderr = io.StringIO()

        with (
            patch("src.sinks.filesink.cv2.VideoWriter", return_value=writer),
            contextlib.redirect_stderr(stderr),
        ):
            FileSink("out", {"path": "out.mp4", "quality": 95}).process(
                {"in": frame}
            )

        self.assertIn("filesink quality option was not accepted", stderr.getvalue())

    def test_filesink_rejects_invalid_quality(self) -> None:
        for quality in (-1, 101):
            with self.subTest(quality=quality), self.assertRaises(ValueError):
                FileSink("out", {"path": "out.mp4", "quality": quality})

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
