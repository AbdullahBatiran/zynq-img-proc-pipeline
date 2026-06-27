"""Difference-of-Gaussians transformer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    apply_per_channel,
    clip_cast_preserve_dtype,
    enhanced_metadata,
    minmax_normalize,
    normalize_aliases,
    parse_bool,
    validate_filter_packet,
)


class Dog(Transformer):
    """Apply Difference-of-Gaussians band-pass enhancement."""

    type_name = "dog"

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
                "sigma-small": ParameterContract("sigma-small", "float", default=1.0),
                "sigma-large": ParameterContract("sigma-large", "float", default=3.0),
                "amount": ParameterContract("amount", "float", default=1.0),
                "bias": ParameterContract("bias", "number", default=0),
                "normalize": ParameterContract("normalize", "bool", default=True),
            },
            description="Enhance band-pass blob detail with Difference of Gaussians.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(
            params, (("sigma_small", "sigma-small"), ("sigma_large", "sigma-large"))
        )
        self.sigma_small = float(normalized.get("sigma-small", 1.0))
        self.sigma_large = float(normalized.get("sigma-large", 3.0))
        self.amount = float(normalized.get("amount", 1.0))
        self.bias = float(normalized.get("bias", 0))
        self.normalize = parse_bool(normalized.get("normalize", True))
        if self.sigma_small <= 0 or self.sigma_large <= 0:
            raise ValueError("dog sigma values must be positive")
        if self.sigma_large <= self.sigma_small:
            raise ValueError("dog sigma-large must be greater than sigma-small")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, self.type_name)
        output = apply_per_channel(packet.data, self._plane)
        metadata = enhanced_metadata(
            packet,
            output,
            self.instance_id,
            self.type_name,
            {
                "sigma-small": self.sigma_small,
                "sigma-large": self.sigma_large,
                "amount": self.amount,
                "bias": self.bias,
                "normalize": self.normalize,
            },
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}

    def _plane(self, plane: np.ndarray) -> np.ndarray:
        working = plane.astype(np.float64)
        small = cv2.GaussianBlur(working, (0, 0), self.sigma_small)
        large = cv2.GaussianBlur(working, (0, 0), self.sigma_large)
        result = (small - large) * self.amount + self.bias
        if self.normalize:
            return minmax_normalize(result, plane.dtype)
        return clip_cast_preserve_dtype(result, plane.dtype)
