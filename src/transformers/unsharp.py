"""Unsharp mask sharpening transformer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    filtered_metadata,
    normalize_aliases,
    validate_filter_packet,
    validate_odd_kernel_size,
    clip_cast_preserve_dtype,
)


class Unsharp(Transformer):
    """Sharpen frames using an unsharp mask."""

    type_name = "unsharp"

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
                "amount": ParameterContract(
                    "amount",
                    "float",
                    default=1.0,
                    description="Unsharp mask strength.",
                ),
                "sigma": ParameterContract(
                    "sigma",
                    "float",
                    default=1.0,
                    description="Gaussian blur sigma for the unsharp mask.",
                ),
                "kernel-size": ParameterContract(
                    "kernel-size",
                    "int",
                    default=0,
                    description="Gaussian kernel size; 0 derives it from sigma.",
                ),
            },
            description="Sharpen frames using an unsharp mask.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(params, (("kernel_size", "kernel-size"),))
        self.amount = float(normalized.get("amount", 1.0))
        self.sigma = float(normalized.get("sigma", 1.0))
        self.kernel_size = int(normalized.get("kernel-size", 0))

        if self.amount < 0:
            raise ValueError("Unsharp amount must be non-negative")
        if self.sigma <= 0:
            raise ValueError("Unsharp sigma must be positive")
        validate_odd_kernel_size(
            self.kernel_size, "Unsharp kernel-size", allow_zero=True
        )

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, "Unsharp")
        ksize = (self.kernel_size, self.kernel_size)
        blurred = cv2.GaussianBlur(packet.data, ksize, self.sigma)
        sharpened = (
            packet.data.astype(np.float64) * (1.0 + self.amount)
            - blurred.astype(np.float64) * self.amount
        )
        sharpened = clip_cast_preserve_dtype(sharpened, packet.data.dtype)
        metadata = filtered_metadata(
            packet,
            sharpened,
            self.instance_id,
            self.type_name,
            {
                "amount": self.amount,
                "sigma": self.sigma,
                "kernel-size": self.kernel_size,
            },
        )
        return {"out": [FramePacket(data=sharpened, metadata=metadata)]}
