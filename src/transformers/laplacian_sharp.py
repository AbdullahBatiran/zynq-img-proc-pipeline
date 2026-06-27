"""Laplacian sharpening transformer."""

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
    validate_odd_kernel_size,
)


_MODES = {"subtract", "add"}


class LaplacianSharp(Transformer):
    """Sharpen frames by repeatedly applying a Laplacian contribution."""

    type_name = "laplacian-sharp"

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
                    description="Laplacian contribution strength.",
                ),
                "kernel-size": ParameterContract(
                    "kernel-size",
                    "int",
                    default=3,
                    description="Laplacian aperture size.",
                ),
                "iterations": ParameterContract(
                    "iterations",
                    "int",
                    default=1,
                    description="Number of Laplacian contributions to apply.",
                ),
                "mode": ParameterContract(
                    "mode",
                    "str",
                    default="subtract",
                    choices=tuple(sorted(_MODES)),
                    description="Subtract or add the Laplacian contribution.",
                ),
                "scale": ParameterContract(
                    "scale",
                    "float",
                    default=1.0,
                    description="Scale passed to cv2.Laplacian.",
                ),
                "delta": ParameterContract(
                    "delta",
                    "float",
                    default=0.0,
                    description="Delta passed to cv2.Laplacian.",
                ),
            },
            description="Sharpen frames with one or more Laplacian passes.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(params, (("kernel_size", "kernel-size"),))
        self.amount = float(normalized.get("amount", 1.0))
        self.kernel_size = int(normalized.get("kernel-size", 3))
        self.iterations = int(normalized.get("iterations", 1))
        self.mode = str(normalized.get("mode", "subtract"))
        self.scale = float(normalized.get("scale", 1.0))
        self.delta = float(normalized.get("delta", 0.0))

        if self.amount < 0:
            raise ValueError("laplacian-sharp amount must be non-negative")
        validate_odd_kernel_size(self.kernel_size, "laplacian-sharp kernel-size")
        if self.iterations <= 0:
            raise ValueError("laplacian-sharp iterations must be positive")
        if self.mode not in _MODES:
            raise ValueError("laplacian-sharp mode must be 'subtract' or 'add'")
        if self.scale < 0:
            raise ValueError("laplacian-sharp scale must be non-negative")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, "laplacian-sharp")
        sharpened = packet.data.astype(np.float64)
        for _ in range(self.iterations):
            laplacian = cv2.Laplacian(
                sharpened,
                cv2.CV_64F,
                ksize=self.kernel_size,
                scale=self.scale,
                delta=self.delta,
            )
            if self.mode == "subtract":
                sharpened = sharpened - self.amount * laplacian
            else:
                sharpened = sharpened + self.amount * laplacian

        sharpened = clip_cast_preserve_dtype(sharpened, packet.data.dtype)
        metadata = filtered_metadata(
            packet,
            sharpened,
            self.instance_id,
            self.type_name,
            {
                "amount": self.amount,
                "kernel-size": self.kernel_size,
                "iterations": self.iterations,
                "mode": self.mode,
                "scale": self.scale,
                "delta": self.delta,
            },
        )
        return {"out": [FramePacket(data=sharpened, metadata=metadata)]}
