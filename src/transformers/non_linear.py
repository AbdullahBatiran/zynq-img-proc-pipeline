"""Non-linear histogram scaling transformer."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket, infer_frame_shape
from src.transformers._filter_utils import (
    normalize_aliases,
    resolve_output_bits_and_max,
)


class NonLinear(Transformer):
    """Apply histogram-based non-linear scaling to gray frames."""

    type_name = "non-linear"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in", formats={"gray"}, depths={8, 16})},
            output_ports={"out": PortContract("out", formats={"gray"}, depths={8, 16})},
            parameters={
                "output-bits": ParameterContract(
                    "output-bits",
                    "int",
                    default="<container depth>",
                    description="Effective output bit depth within the dtype container.",
                ),
            },
            description="Apply histogram-based non-linear scaling.",
            subcategory="Contrast",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(params, (("output_bits", "output-bits"),))
        self.output_bits = (
            int(normalized["output-bits"])
            if normalized.get("output-bits") is not None
            else None
        )
        if self.output_bits is not None and self.output_bits <= 0:
            raise ValueError("non-linear output-bits must be a positive integer")
        if self.output_bits is not None and self.output_bits > 16:
            raise ValueError("non-linear output-bits cannot exceed 16")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        self._validate_packet(packet)
        output_bits, output_max = resolve_output_bits_and_max(
            packet.data.dtype, self.output_bits, self.type_name
        )

        scaled, modified_total = _nonlinear_scale(packet.data, output_max)
        width, height, channels, depth = infer_frame_shape(scaled)
        metadata = packet.metadata.derive(
            width=width,
            height=height,
            channels=channels,
            depth=depth,
            extra={
                **packet.metadata.extra,
                "non_linear_by": self.instance_id,
                "non_linear_output_bits": output_bits,
                "non_linear_output_max": output_max,
                "non_linear_levels": output_max + 1,
                "non_linear_modified_total": modified_total,
            },
        )
        return {"out": [FramePacket(data=scaled, metadata=metadata)]}

    def _validate_packet(self, packet: FramePacket) -> None:
        metadata = packet.metadata
        if metadata.format != "gray":
            raise ValueError("non-linear supports only gray frames")
        if metadata.depth not in {8, 16}:
            raise ValueError("non-linear supports only 8-bit and 16-bit frames")
        if metadata.channels != 1:
            raise ValueError("non-linear supports only one-channel frames")
        expected_dtype = np.uint8 if metadata.depth == 8 else np.uint16
        if packet.data.dtype != expected_dtype:
            raise ValueError(
                f"non-linear expected dtype {expected_dtype} for "
                f"{metadata.depth}-bit metadata"
            )
        if packet.data.ndim == 2:
            return
        if packet.data.ndim == 3 and packet.data.shape[2] == 1:
            return
        raise ValueError("non-linear input must be 2D or HxWx1")


def _nonlinear_scale(frame: np.ndarray, output_max: int) -> tuple[np.ndarray, int]:
    clipped = np.clip(frame, 0, output_max).astype(np.int64, copy=False)
    levels = output_max + 1
    histogram = np.bincount(clipped.ravel(), minlength=levels).astype(
        np.uint64, copy=False
    )

    modified = np.zeros_like(histogram, dtype=np.uint64)
    nonzero = histogram > 0
    modified[nonzero] = np.floor(
        np.log2(histogram[nonzero].astype(np.float64))
    ).astype(np.uint64)
    cumulative = np.cumsum(modified, dtype=np.uint64)

    modified_total = int(cumulative[-1]) if cumulative.size else 0
    if modified_total == 0:
        lut = np.zeros(levels, dtype=frame.dtype)
    else:
        lut_float = cumulative.astype(np.float64) * (float(output_max) / modified_total)
        lut = np.rint(np.clip(lut_float, 0, output_max)).astype(frame.dtype)

    return lut[clipped].astype(frame.dtype, copy=False), modified_total
