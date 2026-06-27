"""Median filter transformer."""

from __future__ import annotations

from typing import Any

import cv2

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    filtered_metadata,
    normalize_aliases,
    validate_filter_packet,
    validate_odd_kernel_size,
)


class Median(Transformer):
    """Apply a median filter while preserving frame dtype."""

    type_name = "median"

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
                "kernel-size": ParameterContract(
                    "kernel-size",
                    "int",
                    default=3,
                    description="Median filter aperture size.",
                ),
            },
            description="Apply a median filter.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(params, (("kernel_size", "kernel-size"),))
        self.kernel_size = int(normalized.get("kernel-size", 3))
        validate_odd_kernel_size(self.kernel_size, "median kernel-size")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, "median")
        if packet.metadata.depth == 16 and self.kernel_size not in {3, 5}:
            raise ValueError("median uint16 kernel-size must be 3 or 5")

        filtered = cv2.medianBlur(packet.data, self.kernel_size)
        metadata = filtered_metadata(
            packet,
            filtered,
            self.instance_id,
            self.type_name,
            {"kernel-size": self.kernel_size},
        )
        return {"out": [FramePacket(data=filtered, metadata=metadata)]}
