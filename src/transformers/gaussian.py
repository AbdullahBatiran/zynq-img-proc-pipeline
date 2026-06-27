"""Gaussian filter transformer."""

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


class Gaussian(Transformer):
    """Apply a Gaussian blur while preserving frame dtype."""

    type_name = "gaussian"

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
                    default=5,
                    description="Gaussian kernel size.",
                ),
                "sigma-x": ParameterContract(
                    "sigma-x",
                    "float",
                    default=0.0,
                    description="Gaussian sigma in X; 0 lets OpenCV derive it.",
                ),
                "sigma-y": ParameterContract(
                    "sigma-y",
                    "float",
                    default=0.0,
                    description="Gaussian sigma in Y; 0 lets OpenCV derive it.",
                ),
            },
            description="Apply a Gaussian blur.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(
            params,
            (
                ("kernel_size", "kernel-size"),
                ("sigma_x", "sigma-x"),
                ("sigma_y", "sigma-y"),
            ),
        )
        self.kernel_size = int(normalized.get("kernel-size", 5))
        self.sigma_x = float(normalized.get("sigma-x", 0.0))
        self.sigma_y = float(normalized.get("sigma-y", 0.0))

        validate_odd_kernel_size(self.kernel_size, "gaussian kernel-size")
        if self.sigma_x < 0:
            raise ValueError("gaussian sigma-x must be non-negative")
        if self.sigma_y < 0:
            raise ValueError("gaussian sigma-y must be non-negative")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, "gaussian")
        filtered = cv2.GaussianBlur(
            packet.data,
            (self.kernel_size, self.kernel_size),
            self.sigma_x,
            sigmaY=self.sigma_y,
        )
        metadata = filtered_metadata(
            packet,
            filtered,
            self.instance_id,
            self.type_name,
            {
                "kernel-size": self.kernel_size,
                "sigma-x": self.sigma_x,
                "sigma-y": self.sigma_y,
            },
        )
        return {"out": [FramePacket(data=filtered, metadata=metadata)]}
