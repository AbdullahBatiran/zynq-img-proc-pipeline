"""Shared helpers for image filter transformers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.packets import FramePacket, infer_frame_shape


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


def dtype_max(dtype: np.dtype) -> int:
    if dtype == np.dtype(np.uint8):
        return 255
    if dtype == np.dtype(np.uint16):
        return 65535
    raise ValueError("Filters support only uint8 and uint16 frames")


def clip_cast_preserve_dtype(frame: np.ndarray, dtype: np.dtype) -> np.ndarray:
    max_value = dtype_max(dtype)
    clipped = np.clip(frame, 0, max_value)
    if np.issubdtype(dtype, np.integer):
        clipped = np.rint(clipped)
    return clipped.astype(dtype)


def validate_odd_kernel_size(value: int, name: str, *, allow_zero: bool = False) -> None:
    if allow_zero and value == 0:
        return
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
