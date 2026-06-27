"""Shared helpers for image filter transformers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.packets import FrameMetadata, FramePacket, infer_frame_shape


SUPPORTED_FORMATS = {"bgr", "rgb", "gray"}
SUPPORTED_DEPTHS = {8, 16}
SUPPORTED_CHANNELS = {1, 3}


def validate_filter_packet(packet: FramePacket, element_name: str) -> None:
    metadata = packet.metadata
    if metadata.format not in SUPPORTED_FORMATS:
        raise ValueError(f"{element_name} does not support format {metadata.format!r}")
    if metadata.depth not in SUPPORTED_DEPTHS:
        raise ValueError(f"{element_name} supports only 8-bit and 16-bit frames")
    if metadata.channels not in SUPPORTED_CHANNELS:
        raise ValueError(f"{element_name} supports only 1-channel or 3-channel frames")

    expected_dtype = np.uint8 if metadata.depth == 8 else np.uint16
    if packet.data.dtype != expected_dtype:
        raise ValueError(
            f"{element_name} expected dtype {expected_dtype} for "
            f"{metadata.depth}-bit metadata"
        )
    if metadata.channels == 1 and packet.data.ndim not in {2, 3}:
        raise ValueError("1-channel frames must be 2D or HxWx1 arrays")
    if metadata.channels == 3 and (
        packet.data.ndim != 3 or packet.data.shape[2] != 3
    ):
        raise ValueError("3-channel frames must be HxWx3 arrays")


def filtered_metadata(
    packet: FramePacket,
    filtered: np.ndarray,
    instance_id: str,
    filter_name: str,
    params: dict[str, Any],
):
    width, height, channels, depth = infer_frame_shape(filtered)
    return packet.metadata.derive(
        width=width,
        height=height,
        channels=channels,
        depth=depth,
        extra={
            **packet.metadata.extra,
            "filtered_by": instance_id,
            "filter_name": filter_name,
            "filter_params": params,
        },
    )


def enhanced_metadata(
    packet: FramePacket,
    enhanced: np.ndarray,
    instance_id: str,
    enhancement_name: str,
    params: dict[str, Any],
) -> FrameMetadata:
    width, height, channels, depth = infer_frame_shape(enhanced)
    return packet.metadata.derive(
        width=width,
        height=height,
        channels=channels,
        depth=depth,
        extra={
            **packet.metadata.extra,
            "enhanced_by": instance_id,
            "enhancement_name": enhancement_name,
            "enhancement_params": params,
        },
    )


def normalize_aliases(
    params: dict[str, Any], aliases: tuple[tuple[str, str], ...]
) -> dict[str, Any]:
    normalized = dict(params)
    for alias, canonical in aliases:
        if alias in normalized:
            if canonical in normalized:
                raise ValueError(
                    f"Cannot receive both {alias!r} and {canonical!r}"
                )
            normalized[canonical] = normalized.pop(alias)
    return normalized


def parse_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [float(part) for part in parts if part]
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    return [float(value)]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return bool(value)


def dtype_max(dtype: np.dtype) -> int:
    if dtype == np.dtype(np.uint8):
        return 255
    if dtype == np.dtype(np.uint16):
        return 65535
    raise ValueError("Filters support only uint8 and uint16 frames")


def normalized_float(frame: np.ndarray, input_max: float | None = None) -> np.ndarray:
    max_value = float(input_max if input_max is not None else dtype_max(frame.dtype))
    if max_value <= 0:
        raise ValueError("input max must be positive")
    return np.clip(frame.astype(np.float64) / max_value, 0.0, 1.0)


def from_normalized_float(frame: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return clip_cast_preserve_dtype(frame * dtype_max(dtype), dtype)


def minmax_normalize(frame: np.ndarray, dtype: np.dtype) -> np.ndarray:
    working = frame.astype(np.float64)
    min_value = float(np.min(working))
    max_value = float(np.max(working))
    if max_value <= min_value:
        return np.zeros_like(working, dtype=dtype)
    normalized = (working - min_value) / (max_value - min_value)
    return from_normalized_float(normalized, dtype)


def clip_cast_preserve_dtype(frame: np.ndarray, dtype: np.dtype) -> np.ndarray:
    max_value = dtype_max(dtype)
    clipped = np.clip(frame, 0, max_value)
    if np.issubdtype(dtype, np.integer):
        clipped = np.rint(clipped)
    return clipped.astype(dtype)


def apply_per_channel(frame: np.ndarray, func):
    if frame.ndim == 2:
        return func(frame)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return func(frame[:, :, 0])[:, :, np.newaxis]
    channels = [func(frame[:, :, channel]) for channel in range(frame.shape[2])]
    return np.stack(channels, axis=2)


def validate_odd_kernel_size(value: int, name: str, *, allow_zero: bool = False) -> None:
    if allow_zero and value == 0:
        return
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
