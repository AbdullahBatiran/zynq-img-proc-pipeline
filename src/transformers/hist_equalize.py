"""Normal histogram equalization transformer."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket, infer_frame_shape


class HistEqualize(Transformer):
    """Apply normal histogram equalization to 8-bit or 16-bit frames."""

    type_name = "hist_equalize"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={
                "in": PortContract(
                    "in", formats={"bgr", "rgb", "gray"}, depths={8, 16}
                )
            },
            output_ports={"out": PortContract("out", depths={8, 16})},
            description="Apply normal histogram equalization.",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.bins = int(params["bins"]) if params.get("bins") is not None else None
        if self.bins is not None and self.bins <= 0:
            raise ValueError("hist_equalize bins must be a positive integer")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        self._validate_packet(packet)
        bins = self.bins or _default_bins(packet.metadata.depth)
        equalized = self._equalize(packet.data, bins)
        width, height, channels, depth = infer_frame_shape(equalized)
        metadata = packet.metadata.derive(
            width=width,
            height=height,
            channels=channels,
            depth=depth,
            extra={
                **packet.metadata.extra,
                "hist_equalized_by": self.instance_id,
                "hist_bins": bins,
            },
        )
        return {"out": [FramePacket(data=equalized, metadata=metadata)]}

    def _validate_packet(self, packet: FramePacket) -> None:
        metadata = packet.metadata
        if metadata.format not in {"bgr", "rgb", "gray"}:
            raise ValueError(
                f"hist_equalize does not support format {metadata.format!r}"
            )
        if metadata.depth not in {8, 16}:
            raise ValueError("hist_equalize supports only 8-bit and 16-bit frames")
        if metadata.channels not in {1, 3}:
            raise ValueError("hist_equalize supports only 1-channel or 3-channel frames")
        expected_dtype = np.uint8 if metadata.depth == 8 else np.uint16
        if packet.data.dtype != expected_dtype:
            raise ValueError(
                f"hist_equalize expected dtype {expected_dtype} for "
                f"{metadata.depth}-bit metadata"
            )
        if metadata.channels == 1 and packet.data.ndim not in {2, 3}:
            raise ValueError("1-channel frames must be 2D or HxWx1 arrays")
        if metadata.channels == 3 and (
            packet.data.ndim != 3 or packet.data.shape[2] != 3
        ):
            raise ValueError("3-channel frames must be HxWx3 arrays")

    def _equalize(self, frame: np.ndarray, bins: int) -> np.ndarray:
        if frame.ndim == 2:
            return _equalize_plane(frame, bins)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return _equalize_plane(frame[:, :, 0], bins)[:, :, np.newaxis]
        channels = [_equalize_plane(frame[:, :, channel], bins) for channel in range(3)]
        return np.stack(channels, axis=2)


def _equalize_plane(plane: np.ndarray, bins: int) -> np.ndarray:
    max_value = _max_value_for_dtype(plane.dtype)
    hist, _ = np.histogram(plane, bins=bins, range=(0, max_value + 1))
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        return plane.copy()
    cdf_min = int(nonzero[0])
    total = int(cdf[-1])
    if total == cdf_min:
        return plane.copy()

    scale = bins / float(max_value + 1)
    indexes = np.floor(plane.astype(np.float64) * scale).astype(np.int64)
    indexes = np.clip(indexes, 0, bins - 1)
    mapped = np.round((cdf[indexes] - cdf_min) / (total - cdf_min) * max_value)
    return np.clip(mapped, 0, max_value).astype(plane.dtype)


def _default_bins(depth: int) -> int:
    if depth == 8:
        return 256
    if depth == 16:
        return 65536
    raise ValueError("hist_equalize supports only 8-bit and 16-bit frames")


def _max_value_for_dtype(dtype: np.dtype) -> int:
    if dtype == np.dtype(np.uint8):
        return 255
    if dtype == np.dtype(np.uint16):
        return 65535
    raise ValueError("hist_equalize supports only uint8 and uint16 frames")
