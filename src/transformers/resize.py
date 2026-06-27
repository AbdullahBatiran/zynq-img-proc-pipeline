"""Resize transformer."""

from __future__ import annotations

from typing import Any

import cv2

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket, infer_frame_shape


_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


class Resize(Transformer):
    """Resize frames and update frame dimensions in metadata."""

    type_name = "resize"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in")},
            output_ports={"out": PortContract("out")},
            parameters={
                "width": ParameterContract(
                    "width", "int", required=True, description="Output frame width."
                ),
                "height": ParameterContract(
                    "height", "int", required=True, description="Output frame height."
                ),
                "interpolation": ParameterContract(
                    "interpolation",
                    "str",
                    default="linear",
                    choices=tuple(_INTERPOLATION),
                    description="OpenCV resize interpolation mode.",
                ),
            },
            description="Resize video frames.",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.width = int(params["width"])
        self.height = int(params["height"])
        interpolation_name = str(params.get("interpolation", "linear"))
        if interpolation_name not in _INTERPOLATION:
            raise ValueError(f"Unsupported interpolation {interpolation_name!r}")
        self.interpolation = _INTERPOLATION[interpolation_name]

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        resized = cv2.resize(
            packet.data,
            (self.width, self.height),
            interpolation=self.interpolation,
        )
        width, height, channels, depth = infer_frame_shape(resized)
        metadata = packet.metadata.derive(
            width=width,
            height=height,
            channels=channels,
            depth=depth,
            extra={**packet.metadata.extra, "resized_by": self.instance_id},
        )
        return {"out": [FramePacket(data=resized, metadata=metadata)]}
