"""Bit shift transformer."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import validate_filter_packet


_DIRECTIONS = {"left", "right"}


class BitShift(Transformer):
    """Shift every frame component left or right by a fixed bit count."""

    type_name = "bit-shift"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={
                "in": PortContract(
                    "in", formats={"bgr", "rgb", "gray"}, depths={8, 16}
                )
            },
            output_ports={
                "out": PortContract(
                    "out", formats={"bgr", "rgb", "gray"}, depths={8, 16}
                )
            },
            parameters={
                "bits": ParameterContract(
                    "bits",
                    "int",
                    required=True,
                    description="Number of bits to shift.",
                ),
                "direction": ParameterContract(
                    "direction",
                    "str",
                    default="right",
                    choices=tuple(sorted(_DIRECTIONS)),
                    description="Shift direction.",
                ),
            },
            description="Shift every frame component left or right.",
            subcategory="Intensity",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.bits = int(params["bits"])
        if self.bits < 0:
            raise ValueError("bit-shift bits must be non-negative")
        self.direction = str(params.get("direction", "right"))
        if self.direction not in _DIRECTIONS:
            raise ValueError("bit-shift direction must be left or right")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, "bit-shift")

        if self.direction == "left":
            shifted = np.left_shift(packet.data, self.bits)
        else:
            shifted = np.right_shift(packet.data, self.bits)
        shifted = shifted.astype(packet.data.dtype, copy=False)

        metadata = packet.metadata.derive(
            extra={
                **packet.metadata.extra,
                "bit_shifted_by": self.instance_id,
                "bit_shift_bits": self.bits,
                "bit_shift_direction": self.direction,
            }
        )
        return {"out": [FramePacket(data=shifted, metadata=metadata)]}
