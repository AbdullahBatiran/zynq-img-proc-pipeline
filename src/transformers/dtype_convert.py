"""Frame dtype conversion transformer."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket, infer_frame_shape


_DTYPES: dict[str, type[np.unsignedinteger[Any]]] = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
}
_FORMATS = {"bgr", "rgb", "gray"}
_DEPTHS = {8, 16, 32}
_ANSI_YELLOW = "\033[33m"
_ANSI_RESET = "\033[0m"


class DtypeConvert(Transformer):
    """Convert frame dtype without scaling values."""

    type_name = "dtype-convert"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={
                "in": PortContract("in", formats=_FORMATS, depths=_DEPTHS)
            },
            output_ports={
                "out": PortContract("out", formats=_FORMATS, depths=_DEPTHS)
            },
            parameters={
                "dtype": ParameterContract(
                    "dtype",
                    "str",
                    required=True,
                    choices=tuple(_DTYPES),
                    description="Output dtype.",
                ),
            },
            description="Convert frame dtype without scaling values.",
            subcategory="Intensity",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.dtype_name = str(params["dtype"])
        try:
            self.dtype = np.dtype(_DTYPES[self.dtype_name])
        except KeyError as exc:
            raise ValueError(
                "dtype-convert dtype must be uint8, uint16, or uint32"
            ) from exc

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        self._validate_packet(packet)

        converted = self._convert(packet.data)
        width, height, channels, depth = infer_frame_shape(converted)
        metadata = packet.metadata.derive(
            width=width,
            height=height,
            channels=channels,
            depth=depth,
            extra={
                **packet.metadata.extra,
                "dtype_converted_by": self.instance_id,
                "dtype_convert_input_dtype": packet.data.dtype.name,
                "dtype_convert_output_dtype": self.dtype.name,
            },
        )
        return {"out": [FramePacket(data=converted, metadata=metadata)]}

    def _convert(self, frame: np.ndarray) -> np.ndarray:
        target_info = np.iinfo(self.dtype)
        clipped_mask = frame > target_info.max
        if np.any(clipped_mask):
            first_location = tuple(int(index) for index in np.argwhere(clipped_mask)[0])
            first_value = int(frame[first_location])
            print(
                f"{_ANSI_YELLOW}Warning: dtype-convert clipping values above "
                f"{target_info.max} for output dtype {self.dtype.name}; "
                f"first clipped value={first_value} at "
                f"{_format_location(first_location)}{_ANSI_RESET}",
                file=sys.stderr,
            )
            frame = np.clip(frame, target_info.min, target_info.max)
        return frame.astype(self.dtype, copy=True)

    def _validate_packet(self, packet: FramePacket) -> None:
        metadata = packet.metadata
        if metadata.format not in _FORMATS:
            raise ValueError(f"dtype-convert does not support format {metadata.format!r}")
        if metadata.depth not in _DEPTHS:
            raise ValueError(
                "dtype-convert supports only 8-bit, 16-bit, and 32-bit frames"
            )
        if metadata.channels not in {1, 3}:
            raise ValueError("dtype-convert supports only 1-channel or 3-channel frames")
        if packet.data.dtype not in {
            np.dtype(np.uint8),
            np.dtype(np.uint16),
            np.dtype(np.uint32),
        }:
            raise ValueError(
                "dtype-convert supports only uint8, uint16, and uint32 frames"
            )
        expected_dtype = {
            8: np.uint8,
            16: np.uint16,
            32: np.uint32,
        }[metadata.depth]
        if packet.data.dtype != expected_dtype:
            raise ValueError(
                f"dtype-convert expected dtype {expected_dtype} for "
                f"{metadata.depth}-bit metadata"
            )
        if metadata.channels == 1 and packet.data.ndim not in {2, 3}:
            raise ValueError("1-channel frames must be 2D or HxWx1 arrays")
        if metadata.channels == 3 and (
            packet.data.ndim != 3 or packet.data.shape[2] != 3
        ):
            raise ValueError("3-channel frames must be HxWx3 arrays")


def _format_location(location: tuple[int, ...]) -> str:
    if len(location) == 2:
        row, col = location
        return f"row={row}, col={col}"
    if len(location) == 3:
        row, col, channel = location
        return f"row={row}, col={col}, channel={channel}"
    return "index=" + ",".join(str(index) for index in location)
