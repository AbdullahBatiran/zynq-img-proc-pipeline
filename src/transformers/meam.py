"""MEAM base-detail contrast enhancement transformer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket, infer_frame_shape


class Meam(Transformer):
    """Apply MEAM base-detail separation to uint16 mono frames."""

    type_name = "meam"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in", formats={"gray"}, depths={16})},
            output_ports={"out": PortContract("out", formats={"gray"}, depths={8})},
            parameters={
                "detail-gain": ParameterContract(
                    "detail-gain",
                    "float",
                    default=3.0,
                    description="Multiplier for high-frequency detail.",
                ),
                "blur-sigma": ParameterContract(
                    "blur-sigma",
                    "float",
                    default=15.0,
                    description="Gaussian sigma for base/background separation.",
                ),
            },
            description="Apply MEAM base-detail contrast enhancement.",
            subcategory="Contrast",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = dict(params)
        _normalize_alias(normalized, "detail_gain", "detail-gain")
        _normalize_alias(normalized, "blur_sigma", "blur-sigma")
        self.detail_gain = float(normalized.get("detail-gain", 3.0))
        self.blur_sigma = float(normalized.get("blur-sigma", 15.0))
        if self.detail_gain < 0:
            raise ValueError("meam detail-gain must be non-negative")
        if self.blur_sigma <= 0:
            raise ValueError("meam blur-sigma must be positive")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        self._validate_packet(packet)

        enhanced = self._enhance(packet.data)
        width, height, channels, depth = infer_frame_shape(enhanced)
        metadata = packet.metadata.derive(
            width=width,
            height=height,
            channels=channels,
            depth=depth,
            extra={
                **packet.metadata.extra,
                "enhanced_by": self.instance_id,
                "enhancement_name": self.type_name,
                "enhancement_params": {
                    "detail-gain": self.detail_gain,
                    "blur-sigma": self.blur_sigma,
                },
                "meam_base_min": self._last_base_min,
                "meam_base_max": self._last_base_max,
                "meam_dynamic_range": self._last_dynamic_range,
            },
        )
        return {"out": [FramePacket(data=enhanced, metadata=metadata)]}

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            frame = frame[:, :, 0]

        working = frame.astype(np.float32)
        base = cv2.GaussianBlur(working, (0, 0), sigmaX=self.blur_sigma)
        detail = working - base

        base_min = float(np.min(base))
        base_max = float(np.max(base))
        dynamic_range = base_max - base_min
        if dynamic_range == 0:
            dynamic_range = 1.0

        base_8bit = ((base - base_min) / dynamic_range) * 255.0
        detail_8bit = (detail / dynamic_range) * 255.0 * self.detail_gain
        enhanced = base_8bit + detail_8bit

        self._last_base_min = base_min
        self._last_base_max = base_max
        self._last_dynamic_range = dynamic_range
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def _validate_packet(self, packet: FramePacket) -> None:
        metadata = packet.metadata
        if metadata.format != "gray":
            raise ValueError("meam supports only gray frames")
        if metadata.depth != 16:
            raise ValueError("meam supports only 16-bit frames")
        if metadata.channels != 1:
            raise ValueError("meam supports only one-channel frames")
        if packet.data.dtype != np.dtype(np.uint16):
            raise ValueError("meam supports only uint16 frames")
        if packet.data.ndim == 2:
            return
        if packet.data.ndim == 3 and packet.data.shape[2] == 1:
            return
        raise ValueError("meam input must be 2D or HxWx1")


def _normalize_alias(params: dict[str, Any], alias: str, canonical: str) -> None:
    if alias not in params:
        return
    if canonical in params:
        raise ValueError(f"Cannot receive both {alias!r} and {canonical!r}")
    params[canonical] = params.pop(alias)
