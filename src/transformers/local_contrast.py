"""Local background contrast enhancement transformer."""

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
    dtype_max,
    enhanced_metadata,
    minmax_normalize,
    parse_bool,
    validate_filter_packet,
)


_MODES = {"subtract", "divide", "unsharp"}


class LocalContrast(Transformer):
    """Enhance local contrast using a Gaussian background estimate."""

    type_name = "local-contrast"

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
                "mode": ParameterContract(
                    "mode", "str", default="subtract", choices=tuple(sorted(_MODES))
                ),
                "sigma": ParameterContract("sigma", "float", default=15.0),
                "amount": ParameterContract("amount", "float", default=1.0),
                "epsilon": ParameterContract("epsilon", "float", default=1e-6),
                "normalize": ParameterContract("normalize", "bool", default=False),
            },
            description="Enhance local contrast from a blurred background estimate.",
            subcategory="Contrast",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.mode = str(params.get("mode", "subtract"))
        self.sigma = float(params.get("sigma", 15.0))
        self.amount = float(params.get("amount", 1.0))
        self.epsilon = float(params.get("epsilon", 1e-6))
        self.normalize = parse_bool(params.get("normalize", False))
        if self.mode not in _MODES:
            raise ValueError("local-contrast mode must be subtract, divide, or unsharp")
        if self.sigma <= 0:
            raise ValueError("local-contrast sigma must be positive")
        if self.epsilon <= 0:
            raise ValueError("local-contrast epsilon must be positive")

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
                "mode": self.mode,
                "sigma": self.sigma,
                "amount": self.amount,
                "epsilon": self.epsilon,
                "normalize": self.normalize,
            },
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}

    def _plane(self, plane: np.ndarray) -> np.ndarray:
        working = plane.astype(np.float64)
        background = cv2.GaussianBlur(working, (0, 0), self.sigma)
        if self.mode == "subtract":
            result = working + self.amount * (working - background)
        elif self.mode == "divide":
            mean_background = float(np.mean(background))
            result = working * (mean_background + self.epsilon) / (
                background + self.epsilon
            )
        else:
            result = working * (1.0 + self.amount) - background * self.amount
        if self.normalize:
            return minmax_normalize(result, plane.dtype)
        return clip_cast_preserve_dtype(result, plane.dtype)
