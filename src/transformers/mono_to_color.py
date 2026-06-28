"""Mono to color transformer."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket, infer_frame_shape


_FORMATS = {"bgr", "rgb"}


class MonoToColor(Transformer):
    """Replicate mono frames into BGR or RGB color frames."""

    type_name = "mono-to-color"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in", formats={"gray"}, depths={8, 16})},
            output_ports={
                "out": PortContract("out", formats={"bgr", "rgb"}, depths={8, 16})
            },
            parameters={
                "format": ParameterContract(
                    "format",
                    "str",
                    default="bgr",
                    choices=tuple(sorted(_FORMATS)),
                    description="Output color channel order.",
                ),
            },
            description="Convert mono frames to BGR or RGB by channel replication.",
            subcategory="Color",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.format = str(params.get("format", "bgr"))
        if self.format not in _FORMATS:
            raise ValueError("mono-to-color format must be bgr or rgb")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        self._validate_packet(packet)

        if packet.data.ndim == 2:
            mono = packet.data[:, :, np.newaxis]
        else:
            mono = packet.data
        color = np.repeat(mono, 3, axis=2)

        width, height, channels, depth = infer_frame_shape(color)
        metadata = packet.metadata.derive(
            width=width,
            height=height,
            channels=channels,
            depth=depth,
            format=self.format,
            extra={
                **packet.metadata.extra,
                "mono_to_color_by": self.instance_id,
                "mono_to_color_format": self.format,
            },
        )
        return {"out": [FramePacket(data=color, metadata=metadata)]}

    def _validate_packet(self, packet: FramePacket) -> None:
        metadata = packet.metadata
        if metadata.format != "gray":
            raise ValueError("mono-to-color supports only gray frames")
        if metadata.depth not in {8, 16}:
            raise ValueError("mono-to-color supports only 8-bit and 16-bit frames")
        if packet.data.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
            raise ValueError("mono-to-color supports only uint8 and uint16 frames")
        expected_dtype = np.uint8 if metadata.depth == 8 else np.uint16
        if packet.data.dtype != expected_dtype:
            raise ValueError(
                f"mono-to-color expected dtype {expected_dtype} for "
                f"{metadata.depth}-bit metadata"
            )
        if metadata.channels != 1:
            raise ValueError("mono-to-color supports only one-channel frames")
        if packet.data.ndim == 2:
            return
        if packet.data.ndim == 3 and packet.data.shape[2] == 1:
            return
        raise ValueError("mono-to-color input must be 2D or HxWx1")
