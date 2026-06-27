"""Normal histogram equalization transformer."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
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
            parameters={
                "bins": ParameterContract(
                    "bins",
                    "int",
                    default="<output range size>",
                    description="Histogram bin count used for CDF equalization.",
                ),
                "output-bits": ParameterContract(
                    "output-bits",
                    "int",
                    default="<container depth>",
                    description="Effective output bit depth within the dtype container.",
                ),
                "output-max": ParameterContract(
                    "output-max",
                    "int",
                    default="<dtype max>",
                    description="Maximum equalized output value.",
                ),
            },
            description="Apply normal histogram equalization.",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized_params = _normalize_aliases(params)
        self.bins = (
            int(normalized_params["bins"])
            if normalized_params.get("bins") is not None
            else None
        )
        if self.bins is not None and self.bins <= 0:
            raise ValueError("hist_equalize bins must be a positive integer")
        self.output_bits = (
            int(normalized_params["output-bits"])
            if normalized_params.get("output-bits") is not None
            else None
        )
        self.output_max = (
            int(normalized_params["output-max"])
            if normalized_params.get("output-max") is not None
            else None
        )
        if self.output_bits is not None and self.output_bits <= 0:
            raise ValueError("hist_equalize output-bits must be a positive integer")
        if self.output_max is not None and self.output_max < 0:
            raise ValueError("hist_equalize output-max must be non-negative")
        if self.output_bits is not None and self.output_max is not None:
            raise ValueError("hist_equalize cannot combine output-bits and output-max")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        self._validate_packet(packet)
        output_max = self._output_max(packet.data.dtype)
        bins = self.bins or output_max + 1
        equalized = self._equalize(packet.data, bins, output_max)
        width, height, channels, depth = infer_frame_shape(equalized)
        extra = {
            **packet.metadata.extra,
            "hist_equalized_by": self.instance_id,
            "hist_bins": bins,
            "hist_output_max": output_max,
        }
        if self.output_bits is not None:
            extra["hist_output_bits"] = self.output_bits
        metadata = packet.metadata.derive(
            width=width,
            height=height,
            channels=channels,
            depth=depth,
            extra=extra,
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

    def _equalize(self, frame: np.ndarray, bins: int, output_max: int) -> np.ndarray:
        if frame.ndim == 2:
            return _equalize_plane(frame, bins, output_max)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return _equalize_plane(frame[:, :, 0], bins, output_max)[:, :, np.newaxis]
        channels = [
            _equalize_plane(frame[:, :, channel], bins, output_max)
            for channel in range(3)
        ]
        return np.stack(channels, axis=2)

    def _output_max(self, dtype: np.dtype) -> int:
        dtype_max = _max_value_for_dtype(dtype)
        if self.output_bits is not None:
            if self.output_bits > dtype.itemsize * 8:
                raise ValueError(
                    "hist_equalize output-bits cannot exceed container depth"
                )
            return (1 << self.output_bits) - 1
        if self.output_max is not None:
            if self.output_max > dtype_max:
                raise ValueError("hist_equalize output-max cannot exceed dtype max")
            return self.output_max
        return dtype_max


def _equalize_plane(plane: np.ndarray, bins: int, output_max: int) -> np.ndarray:
    working = np.clip(plane, 0, output_max)
    hist, _ = np.histogram(working, bins=bins, range=(0, output_max + 1))
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        return plane.copy()
    cdf_min = int(nonzero[0])
    total = int(cdf[-1])
    if total == cdf_min:
        return working.astype(plane.dtype, copy=True)

    scale = bins / float(output_max + 1)
    indexes = np.floor(working.astype(np.float64) * scale).astype(np.int64)
    indexes = np.clip(indexes, 0, bins - 1)
    mapped = np.round((cdf[indexes] - cdf_min) / (total - cdf_min) * output_max)
    return np.clip(mapped, 0, output_max).astype(plane.dtype)


def _normalize_aliases(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    aliases = (("output_bits", "output-bits"), ("output_max", "output-max"))
    for alias, canonical in aliases:
        if alias in normalized:
            if canonical in normalized:
                raise ValueError(
                    f"hist_equalize cannot receive both {alias!r} and {canonical!r}"
                )
            normalized[canonical] = normalized.pop(alias)
    return normalized


def _max_value_for_dtype(dtype: np.dtype) -> int:
    if dtype == np.dtype(np.uint8):
        return 255
    if dtype == np.dtype(np.uint16):
        return 65535
    raise ValueError("hist_equalize supports only uint8 and uint16 frames")
