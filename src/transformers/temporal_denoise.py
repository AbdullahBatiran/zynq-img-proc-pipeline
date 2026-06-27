"""Stateful temporal denoising transformer."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    clip_cast_preserve_dtype,
    enhanced_metadata,
    validate_filter_packet,
)


_MODES = {"mean", "median", "ema"}


class TemporalDenoise(Transformer):
    """Apply bounded-history temporal denoising."""

    type_name = "temporal-denoise"

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
                "mode": ParameterContract(
                    "mode", "str", default="mean", choices=tuple(sorted(_MODES))
                ),
                "window": ParameterContract("window", "int", default=5),
                "alpha": ParameterContract("alpha", "float", default=0.2),
            },
            description="Apply bounded-history temporal denoising.",
            subcategory="Control",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.mode = str(params.get("mode", "mean"))
        self.window = int(params.get("window", 5))
        self.alpha = float(params.get("alpha", 0.2))
        if self.mode not in _MODES:
            raise ValueError("temporal-denoise mode must be mean, median, or ema")
        if self.window <= 0:
            raise ValueError("temporal-denoise window must be positive")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("temporal-denoise alpha must be in the range (0, 1]")
        self.history: deque[np.ndarray] = deque(maxlen=self.window)
        self.ema: np.ndarray | None = None

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, self.type_name)
        frame = packet.data.astype(np.float64)
        if self.mode == "ema":
            self.ema = frame if self.ema is None else self.alpha * frame + (1.0 - self.alpha) * self.ema
            output = clip_cast_preserve_dtype(self.ema, packet.data.dtype)
            history_size = 1
        else:
            self.history.append(frame.copy())
            stack = np.stack(list(self.history), axis=0)
            if self.mode == "mean":
                filtered = np.mean(stack, axis=0)
            else:
                filtered = np.median(stack, axis=0)
            output = clip_cast_preserve_dtype(filtered, packet.data.dtype)
            history_size = len(self.history)
        metadata = enhanced_metadata(
            packet,
            output,
            self.instance_id,
            self.type_name,
            {"mode": self.mode, "window": self.window, "alpha": self.alpha},
        )
        metadata.extra["temporal_history_size"] = history_size
        return {"out": [FramePacket(data=output, metadata=metadata)]}
