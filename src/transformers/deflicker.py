"""Temporal deflicker transformer."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    clip_cast_preserve_dtype,
    enhanced_metadata,
    normalize_aliases,
    validate_filter_packet,
)


_MODES = {"mean-std", "percentile"}
_TARGETS = {"running"}


class Deflicker(Transformer):
    """Stabilize frame-to-frame global intensity changes."""

    type_name = "deflicker"

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
                    "mode", "str", default="mean-std", choices=tuple(sorted(_MODES))
                ),
                "alpha": ParameterContract("alpha", "float", default=0.1),
                "low-perc": ParameterContract("low-perc", "float", default=0.01),
                "high-perc": ParameterContract("high-perc", "float", default=0.99),
                "target": ParameterContract(
                    "target", "str", default="running", choices=tuple(sorted(_TARGETS))
                ),
            },
            description="Stabilize global brightness/range over time.",
            subcategory="Control",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(
            params, (("low_perc", "low-perc"), ("high_perc", "high-perc"))
        )
        self.mode = str(normalized.get("mode", "mean-std"))
        self.alpha = float(normalized.get("alpha", 0.1))
        self.low_perc = float(normalized.get("low-perc", 0.01))
        self.high_perc = float(normalized.get("high-perc", 0.99))
        self.target = str(normalized.get("target", "running"))
        if self.mode not in _MODES:
            raise ValueError("deflicker mode must be mean-std or percentile")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("deflicker alpha must be in the range (0, 1]")
        if not 0.0 <= self.low_perc < self.high_perc <= 1.0:
            raise ValueError("deflicker percentiles must satisfy 0 <= low < high <= 1")
        if self.target not in _TARGETS:
            raise ValueError("deflicker target must be running")
        self.running: tuple[float, float] | None = None

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, self.type_name)
        frame = packet.data.astype(np.float64)
        current = self._stats(frame)
        if self.running is None:
            self.running = current
        else:
            self.running = (
                (1.0 - self.alpha) * self.running[0] + self.alpha * current[0],
                (1.0 - self.alpha) * self.running[1] + self.alpha * current[1],
            )
        output = self._remap(frame, current, self.running, packet.data.dtype)
        metadata = enhanced_metadata(
            packet,
            output,
            self.instance_id,
            self.type_name,
            {
                "mode": self.mode,
                "alpha": self.alpha,
                "low-perc": self.low_perc,
                "high-perc": self.high_perc,
                "target": self.target,
            },
        )
        metadata.extra["deflicker_current_low"] = current[0]
        metadata.extra["deflicker_current_high"] = current[1]
        metadata.extra["deflicker_target_low"] = self.running[0]
        metadata.extra["deflicker_target_high"] = self.running[1]
        return {"out": [FramePacket(data=output, metadata=metadata)]}

    def _stats(self, frame: np.ndarray) -> tuple[float, float]:
        if self.mode == "mean-std":
            mean = float(np.mean(frame))
            std = float(np.std(frame))
            return mean, max(std, 1e-6)
        return (
            float(np.quantile(frame, self.low_perc)),
            float(np.quantile(frame, self.high_perc)),
        )

    def _remap(
        self,
        frame: np.ndarray,
        current: tuple[float, float],
        target: tuple[float, float],
        dtype: np.dtype,
    ) -> np.ndarray:
        if self.mode == "mean-std":
            result = (frame - current[0]) * (target[1] / max(current[1], 1e-6)) + target[0]
        else:
            result = (frame - current[0]) * (
                (target[1] - target[0]) / max(current[1] - current[0], 1e-6)
            ) + target[0]
        return clip_cast_preserve_dtype(result, dtype)
