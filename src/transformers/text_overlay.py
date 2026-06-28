"""Text overlay transformer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket


_POSITIONS = {
    "top-left",
    "top",
    "top-right",
    "left",
    "center",
    "right",
    "bottom-left",
    "bottom",
    "bottom-right",
}

_FONTS = {
    "simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "plain": cv2.FONT_HERSHEY_PLAIN,
    "duplex": cv2.FONT_HERSHEY_DUPLEX,
    "complex": cv2.FONT_HERSHEY_COMPLEX,
    "triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "complex-small": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "script-simplex": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "script-complex": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
}

_LINE_TYPES = {
    "aa": cv2.LINE_AA,
    "8": cv2.LINE_8,
    "4": cv2.LINE_4,
}

_NAMED_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}


class TextOverlay(Transformer):
    """Draw static text on top of video frames."""

    type_name = "text-overlay"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={
                "in": PortContract(
                    "in",
                    formats={"gray", "bgr", "rgb"},
                    depths={8, 16},
                )
            },
            output_ports={
                "out": PortContract(
                    "out",
                    formats={"gray", "bgr", "rgb"},
                    depths={8, 16},
                )
            },
            parameters={
                "text": ParameterContract(
                    "text", "str", required=True, description="Text to draw."
                ),
                "color": ParameterContract(
                    "color",
                    "str",
                    default="white",
                    description="Named, hex #RRGGBB, or comma-separated RGB color.",
                ),
                "position": ParameterContract(
                    "position",
                    "str",
                    default="top-left",
                    choices=tuple(sorted(_POSITIONS)),
                    description="Anchor position for the text box.",
                ),
                "x": ParameterContract(
                    "x", "int", default=0, description="Horizontal pixel offset."
                ),
                "y": ParameterContract(
                    "y", "int", default=0, description="Vertical pixel offset."
                ),
                "font-size": ParameterContract(
                    "font-size",
                    "float",
                    default=1.0,
                    description="OpenCV font scale.",
                ),
                "thickness": ParameterContract(
                    "thickness",
                    "int",
                    default=1,
                    description="Text stroke thickness.",
                ),
                "font": ParameterContract(
                    "font",
                    "str",
                    default="simplex",
                    choices=tuple(sorted(_FONTS)),
                    description="OpenCV Hershey font face.",
                ),
                "line-type": ParameterContract(
                    "line-type",
                    "str",
                    default="aa",
                    choices=tuple(_LINE_TYPES),
                    description="OpenCV line type.",
                ),
            },
            description="Draw configurable text on top of video frames.",
            subcategory="Compose",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.text = str(params["text"])
        if not self.text:
            raise ValueError("text-overlay text must be non-empty")

        self.color_rgb = _parse_color(params.get("color", "white"))
        self.position = str(params.get("position", "top-left"))
        if self.position not in _POSITIONS:
            raise ValueError(f"Unsupported text-overlay position {self.position!r}")

        self.x = int(params.get("x", 0))
        self.y = int(params.get("y", 0))
        self.font_size = float(params.get("font-size", 1.0))
        if self.font_size <= 0:
            raise ValueError("text-overlay font-size must be positive")

        self.thickness = int(params.get("thickness", 1))
        if self.thickness <= 0:
            raise ValueError("text-overlay thickness must be positive")

        self.font_name = str(params.get("font", "simplex"))
        try:
            self.font = _FONTS[self.font_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported text-overlay font {self.font_name!r}") from exc

        self.line_type_name = str(params.get("line-type", "aa"))
        try:
            self.line_type = _LINE_TYPES[self.line_type_name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported text-overlay line-type {self.line_type_name!r}"
            ) from exc

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        self._validate_packet(packet)

        frame = packet.data.copy()
        color = _color_for_frame(self.color_rgb, packet.metadata.format, frame.dtype)
        origin = self._origin(packet.metadata.width, packet.metadata.height)
        cv2.putText(
            frame,
            self.text,
            origin,
            self.font,
            self.font_size,
            color,
            self.thickness,
            self.line_type,
        )

        metadata = packet.metadata.derive(
            extra={
                **packet.metadata.extra,
                "text_overlay_by": self.instance_id,
                "text_overlay_text": self.text,
                "text_overlay_position": self.position,
                "text_overlay_offset": (self.x, self.y),
                "text_overlay_color": self.color_rgb,
                "text_overlay_font_size": self.font_size,
            }
        )
        return {"out": [FramePacket(data=frame, metadata=metadata)]}

    def _origin(self, frame_width: int, frame_height: int) -> tuple[int, int]:
        (text_width, text_height), baseline = cv2.getTextSize(
            self.text, self.font, self.font_size, self.thickness
        )
        total_height = text_height + baseline

        if self.position.endswith("left"):
            x = 0
        elif self.position.endswith("right"):
            x = frame_width - text_width
        else:
            x = (frame_width - text_width) // 2

        if self.position.startswith("top"):
            y = text_height
        elif self.position.startswith("bottom"):
            y = frame_height - baseline
        else:
            y = (frame_height - total_height) // 2 + text_height

        return (x + self.x, y + self.y)

    def _validate_packet(self, packet: FramePacket) -> None:
        metadata = packet.metadata
        if metadata.format not in {"gray", "bgr", "rgb"}:
            raise ValueError("text-overlay supports only gray, bgr, and rgb frames")
        if packet.data.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            raise ValueError("text-overlay supports only uint8 and uint16 frames")
        if metadata.depth not in {8, 16}:
            raise ValueError("text-overlay supports only 8-bit and 16-bit frames")
        if metadata.format == "gray":
            if metadata.channels != 1 or packet.data.ndim not in {2, 3}:
                raise ValueError("gray text-overlay input must have one channel")
            if packet.data.ndim == 3 and packet.data.shape[2] != 1:
                raise ValueError("gray text-overlay input must have one channel")
            return
        if metadata.channels != 3 or packet.data.ndim != 3 or packet.data.shape[2] != 3:
            raise ValueError("color text-overlay input must have three channels")


def _parse_color(value: Any) -> tuple[int, int, int]:
    text = str(value).strip()
    named = _NAMED_COLORS.get(text.lower())
    if named is not None:
        return named

    if text.startswith("#"):
        if len(text) != 7:
            raise ValueError("text-overlay hex colors must use #RRGGBB")
        try:
            return (
                int(text[1:3], 16),
                int(text[3:5], 16),
                int(text[5:7], 16),
            )
        except ValueError as exc:
            raise ValueError("text-overlay hex colors must use #RRGGBB") from exc

    parts = text.split(",")
    if len(parts) == 3:
        try:
            red, green, blue = (int(part.strip()) for part in parts)
        except ValueError as exc:
            raise ValueError("text-overlay RGB colors must be r,g,b integers") from exc
        color = (red, green, blue)
        if all(0 <= component <= 255 for component in color):
            return color
        raise ValueError("text-overlay RGB color components must be in 0..255")

    raise ValueError(f"Unsupported text-overlay color {value!r}")


def _color_for_frame(
    color_rgb: tuple[int, int, int], frame_format: str, dtype: np.dtype
) -> int | tuple[int, int, int]:
    if dtype == np.dtype(np.uint16):
        color_rgb = tuple(component * 257 for component in color_rgb)

    if frame_format == "gray":
        red, green, blue = color_rgb
        return int(round(0.299 * red + 0.587 * green + 0.114 * blue))
    if frame_format == "bgr":
        red, green, blue = color_rgb
        return (blue, green, red)
    return color_rgb
