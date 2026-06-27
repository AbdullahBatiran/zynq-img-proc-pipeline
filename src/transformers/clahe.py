"""Contrast-limited adaptive histogram equalization transformer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    apply_per_channel,
    dtype_max,
    enhanced_metadata,
    normalize_aliases,
    validate_filter_packet,
)


_CHANNEL_MODES = {"per-channel", "luma"}


class Clahe(Transformer):
    """Apply CLAHE contrast enhancement while preserving dtype."""

    type_name = "clahe"

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
                "clip-limit": ParameterContract(
                    "clip-limit", "float", default=2.0, description="CLAHE clip limit."
                ),
                "tile-grid-size": ParameterContract(
                    "tile-grid-size",
                    "int",
                    default=8,
                    description="Square CLAHE tile grid size.",
                ),
                "bins": ParameterContract(
                    "bins",
                    "int",
                    default="<dtype range size>",
                    description="Recorded histogram bin count.",
                ),
                "channel-mode": ParameterContract(
                    "channel-mode",
                    "str",
                    default="per-channel",
                    choices=tuple(sorted(_CHANNEL_MODES)),
                    description="Apply per channel or enhance luminance.",
                ),
            },
            description="Apply contrast-limited adaptive histogram equalization.",
            subcategory="Contrast",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(
            params,
            (
                ("clip_limit", "clip-limit"),
                ("tile_grid_size", "tile-grid-size"),
                ("channel_mode", "channel-mode"),
            ),
        )
        self.clip_limit = float(normalized.get("clip-limit", 2.0))
        self.tile_grid_size = int(normalized.get("tile-grid-size", 8))
        self.bins = int(normalized["bins"]) if normalized.get("bins") else None
        self.channel_mode = str(normalized.get("channel-mode", "per-channel"))
        if self.clip_limit <= 0:
            raise ValueError("clahe clip-limit must be positive")
        if self.tile_grid_size <= 0:
            raise ValueError("clahe tile-grid-size must be positive")
        if self.bins is not None and self.bins <= 0:
            raise ValueError("clahe bins must be positive")
        if self.channel_mode not in _CHANNEL_MODES:
            raise ValueError("clahe channel-mode must be per-channel or luma")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, self.type_name)
        bins = self.bins or dtype_max(packet.data.dtype) + 1
        if self.channel_mode == "luma" and packet.metadata.channels == 3:
            enhanced = self._luma(packet.data)
        else:
            enhanced = apply_per_channel(packet.data, self._plane)
        metadata = enhanced_metadata(
            packet,
            enhanced,
            self.instance_id,
            self.type_name,
            {
                "clip-limit": self.clip_limit,
                "tile-grid-size": self.tile_grid_size,
                "bins": bins,
                "channel-mode": self.channel_mode,
            },
        )
        return {"out": [FramePacket(data=enhanced, metadata=metadata)]}

    def _plane(self, plane: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_grid_size, self.tile_grid_size),
        )
        return clahe.apply(plane)

    def _luma(self, frame: np.ndarray) -> np.ndarray:
        weights = np.array([0.114, 0.587, 0.299], dtype=np.float64)
        luma = np.sum(frame.astype(np.float64) * weights, axis=2)
        enhanced_luma = self._plane(np.rint(luma).astype(frame.dtype)).astype(np.float64)
        ratio = enhanced_luma / np.maximum(luma, 1.0)
        result = frame.astype(np.float64) * ratio[:, :, np.newaxis]
        return np.clip(np.rint(result), 0, dtype_max(frame.dtype)).astype(frame.dtype)
