"""Guided filter transformer."""

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
    from_normalized_float,
    minmax_normalize,
    normalized_float,
    validate_filter_packet,
)


_MODES = {"smooth", "detail", "enhance"}


class GuidedFilter(Transformer):
    """Apply self-guided edge-preserving filtering."""

    type_name = "guided-filter"

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
                "radius": ParameterContract("radius", "int", default=8),
                "eps": ParameterContract("eps", "float", default=0.01),
                "mode": ParameterContract("mode", "str", default="smooth", choices=tuple(sorted(_MODES))),
                "amount": ParameterContract("amount", "float", default=1.0),
            },
            description="Apply self-guided smoothing or detail enhancement.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.radius = int(params.get("radius", 8))
        self.eps = float(params.get("eps", 0.01))
        self.mode = str(params.get("mode", "smooth"))
        self.amount = float(params.get("amount", 1.0))
        if self.radius <= 0:
            raise ValueError("guided-filter radius must be positive")
        if self.eps <= 0:
            raise ValueError("guided-filter eps must be positive")
        if self.mode not in _MODES:
            raise ValueError("guided-filter mode must be smooth, detail, or enhance")

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
                "radius": self.radius,
                "eps": self.eps,
                "mode": self.mode,
                "amount": self.amount,
            },
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}

    def _plane(self, plane: np.ndarray) -> np.ndarray:
        p = normalized_float(plane)
        q = _guided_plane(p, self.radius, self.eps)
        if self.mode == "smooth":
            return from_normalized_float(q, plane.dtype)
        detail = p - q
        if self.mode == "detail":
            return minmax_normalize(detail, plane.dtype)
        result = p + self.amount * detail
        return clip_cast_preserve_dtype(result * np.iinfo(plane.dtype).max, plane.dtype)


def _box_filter(frame: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.boxFilter(frame, cv2.CV_64F, (size, size), normalize=True)


def _guided_plane(guide: np.ndarray, radius: int, eps: float) -> np.ndarray:
    mean_i = _box_filter(guide, radius)
    mean_p = mean_i
    corr_i = _box_filter(guide * guide, radius)
    corr_ip = corr_i
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)
