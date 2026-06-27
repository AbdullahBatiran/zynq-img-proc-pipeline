"""Morphological image operation transformer."""

from __future__ import annotations

from typing import Any

import cv2

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    enhanced_metadata,
    normalize_aliases,
    validate_filter_packet,
)


_OPS = {
    "erode",
    "dilate",
    "open",
    "close",
    "gradient",
    "tophat",
    "blackhat",
}
_SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}
_MORPH_OPS = {
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat": cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}


class Morphology(Transformer):
    """Apply common morphology operations."""

    type_name = "morphology"

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
                "op": ParameterContract("op", "str", default="open", choices=tuple(sorted(_OPS))),
                "kernel-size": ParameterContract("kernel-size", "int", default=5),
                "kernel-shape": ParameterContract(
                    "kernel-shape", "str", default="rect", choices=tuple(sorted(_SHAPES))
                ),
                "iterations": ParameterContract("iterations", "int", default=1),
            },
            description="Apply morphology operations for shape and target enhancement.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(
            params,
            (("kernel_size", "kernel-size"), ("kernel_shape", "kernel-shape")),
        )
        self.op = str(normalized.get("op", "open"))
        self.kernel_size = int(normalized.get("kernel-size", 5))
        self.kernel_shape = str(normalized.get("kernel-shape", "rect"))
        self.iterations = int(normalized.get("iterations", 1))
        if self.op not in _OPS:
            raise ValueError("morphology op is invalid")
        if self.kernel_size <= 0:
            raise ValueError("morphology kernel-size must be positive")
        if self.kernel_shape not in _SHAPES:
            raise ValueError("morphology kernel-shape is invalid")
        if self.iterations <= 0:
            raise ValueError("morphology iterations must be positive")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, self.type_name)
        kernel = cv2.getStructuringElement(
            _SHAPES[self.kernel_shape], (self.kernel_size, self.kernel_size)
        )
        if self.op == "erode":
            output = cv2.erode(packet.data, kernel, iterations=self.iterations)
        elif self.op == "dilate":
            output = cv2.dilate(packet.data, kernel, iterations=self.iterations)
        else:
            output = cv2.morphologyEx(
                packet.data, _MORPH_OPS[self.op], kernel, iterations=self.iterations
            )
        metadata = enhanced_metadata(
            packet,
            output,
            self.instance_id,
            self.type_name,
            {
                "op": self.op,
                "kernel-size": self.kernel_size,
                "kernel-shape": self.kernel_shape,
                "iterations": self.iterations,
            },
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}
