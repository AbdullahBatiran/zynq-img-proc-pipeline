"""Edge enhancement transformer."""

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
    parse_bool,
    validate_filter_packet,
    validate_odd_kernel_size,
)


_OPERATORS = {"sobel", "scharr"}
_MODES = {"magnitude", "blend"}


class EdgeEnhance(Transformer):
    """Enhance Sobel or Scharr edge magnitude."""

    type_name = "edge-enhance"

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
                "operator": ParameterContract("operator", "str", default="sobel", choices=tuple(sorted(_OPERATORS))),
                "ksize": ParameterContract("ksize", "int", default=3),
                "amount": ParameterContract("amount", "float", default=1.0),
                "mode": ParameterContract("mode", "str", default="magnitude", choices=tuple(sorted(_MODES))),
                "normalize": ParameterContract("normalize", "bool", default=True),
            },
            description="Enhance Sobel or Scharr edge response.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.operator = str(params.get("operator", "sobel"))
        self.ksize = int(params.get("ksize", 3))
        self.amount = float(params.get("amount", 1.0))
        self.mode = str(params.get("mode", "magnitude"))
        self.normalize = parse_bool(params.get("normalize", True))
        if self.operator not in _OPERATORS:
            raise ValueError("edge-enhance operator must be sobel or scharr")
        if self.operator == "sobel":
            validate_odd_kernel_size(self.ksize, "edge-enhance ksize")
        if self.mode not in _MODES:
            raise ValueError("edge-enhance mode must be magnitude or blend")

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
                "operator": self.operator,
                "ksize": self.ksize,
                "amount": self.amount,
                "mode": self.mode,
                "normalize": self.normalize,
            },
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}

    def _plane(self, plane: np.ndarray) -> np.ndarray:
        working = plane.astype(np.float64)
        if self.operator == "scharr":
            gx = cv2.Scharr(working, cv2.CV_64F, 1, 0)
            gy = cv2.Scharr(working, cv2.CV_64F, 0, 1)
        else:
            gx = cv2.Sobel(working, cv2.CV_64F, 1, 0, ksize=self.ksize)
            gy = cv2.Sobel(working, cv2.CV_64F, 0, 1, ksize=self.ksize)
        magnitude = np.sqrt(gx * gx + gy * gy)
        response = minmax_normalize(magnitude, plane.dtype).astype(np.float64)
        if self.mode == "magnitude":
            return response.astype(plane.dtype)
        result = working + self.amount * response
        if self.normalize:
            return minmax_normalize(result, plane.dtype)
        return clip_cast_preserve_dtype(result, plane.dtype)
