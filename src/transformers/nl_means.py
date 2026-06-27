"""Non-local means denoising transformer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.restoration import denoise_nl_means

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    apply_per_channel,
    enhanced_metadata,
    from_normalized_float,
    normalize_aliases,
    normalized_float,
    parse_bool,
    validate_filter_packet,
)


class NlMeans(Transformer):
    """Apply non-local means denoising."""

    type_name = "nl-means"

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
                "h": ParameterContract("h", "float", default=10.0),
                "template-window-size": ParameterContract("template-window-size", "int", default=7),
                "search-window-size": ParameterContract("search-window-size", "int", default=21),
                "fast": ParameterContract("fast", "bool", default=True),
            },
            description="Apply non-local means denoising.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(
            params,
            (
                ("template_window_size", "template-window-size"),
                ("search_window_size", "search-window-size"),
            ),
        )
        self.h = float(normalized.get("h", 10.0))
        self.template_window_size = int(normalized.get("template-window-size", 7))
        self.search_window_size = int(normalized.get("search-window-size", 21))
        self.fast = parse_bool(normalized.get("fast", True))
        if self.h <= 0:
            raise ValueError("nl-means h must be positive")
        if self.template_window_size <= 0 or self.search_window_size <= 0:
            raise ValueError("nl-means window sizes must be positive")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, self.type_name)
        if packet.data.dtype == np.dtype(np.uint8):
            output = apply_per_channel(packet.data, self._plane_uint8)
        else:
            output = apply_per_channel(packet.data, self._plane_uint16)
        metadata = enhanced_metadata(
            packet,
            output,
            self.instance_id,
            self.type_name,
            {
                "h": self.h,
                "template-window-size": self.template_window_size,
                "search-window-size": self.search_window_size,
                "fast": self.fast,
            },
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}

    def _plane_uint8(self, plane: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(
            plane,
            None,
            self.h,
            self.template_window_size,
            self.search_window_size,
        )

    def _plane_uint16(self, plane: np.ndarray) -> np.ndarray:
        normalized = normalized_float(plane)
        denoised = denoise_nl_means(
            normalized,
            h=self.h / 65535.0,
            patch_size=self.template_window_size,
            patch_distance=max(1, self.search_window_size // 2),
            fast_mode=self.fast,
            channel_axis=None,
        )
        return from_normalized_float(denoised, plane.dtype)
