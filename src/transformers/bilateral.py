"""Bilateral filter transformer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    clip_cast_preserve_dtype,
    filtered_metadata,
    normalize_aliases,
    validate_filter_packet,
)


class Bilateral(Transformer):
    """Apply a bilateral filter while preserving frame dtype."""

    type_name = "bilateral"

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
                "diameter": ParameterContract(
                    "diameter",
                    "int",
                    default=5,
                    description="Pixel neighborhood diameter.",
                ),
                "sigma-color": ParameterContract(
                    "sigma-color",
                    "float",
                    default=75.0,
                    description="Filter sigma in color space.",
                ),
                "sigma-space": ParameterContract(
                    "sigma-space",
                    "float",
                    default=75.0,
                    description="Filter sigma in coordinate space.",
                ),
            },
            description="Apply a bilateral filter.",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(
            params,
            (("sigma_color", "sigma-color"), ("sigma_space", "sigma-space")),
        )
        self.diameter = int(normalized.get("diameter", 5))
        self.sigma_color = float(normalized.get("sigma-color", 75.0))
        self.sigma_space = float(normalized.get("sigma-space", 75.0))

        if self.diameter <= 0:
            raise ValueError("bilateral diameter must be positive")
        if self.sigma_color < 0:
            raise ValueError("bilateral sigma-color must be non-negative")
        if self.sigma_space < 0:
            raise ValueError("bilateral sigma-space must be non-negative")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, "bilateral")
        if packet.data.dtype == np.dtype(np.uint8):
            filtered = cv2.bilateralFilter(
                packet.data, self.diameter, self.sigma_color, self.sigma_space
            )
        else:
            filtered_float = cv2.bilateralFilter(
                packet.data.astype(np.float32),
                self.diameter,
                self.sigma_color,
                self.sigma_space,
            )
            filtered = clip_cast_preserve_dtype(filtered_float, packet.data.dtype)

        metadata = filtered_metadata(
            packet,
            filtered,
            self.instance_id,
            self.type_name,
            {
                "diameter": self.diameter,
                "sigma-color": self.sigma_color,
                "sigma-space": self.sigma_space,
            },
        )
        return {"out": [FramePacket(data=filtered, metadata=metadata)]}
