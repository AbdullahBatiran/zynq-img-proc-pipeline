"""Tone-curve contrast transformer."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    enhanced_metadata,
    from_normalized_float,
    normalize_aliases,
    normalized_float,
    parse_bool,
    validate_filter_packet,
)


_MODES = {"gamma", "log", "sigmoid"}


class ToneCurve(Transformer):
    """Apply gamma, log, or sigmoid tone correction."""

    type_name = "tone-curve"

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
                    "mode", "str", default="gamma", choices=tuple(sorted(_MODES))
                ),
                "gamma": ParameterContract("gamma", "float", default=1.0),
                "gain": ParameterContract("gain", "float", default=1.0),
                "cutoff": ParameterContract("cutoff", "float", default=0.5),
                "inverse": ParameterContract("inverse", "bool", default=False),
                "input-max": ParameterContract(
                    "input-max", "number", default="<dtype max>"
                ),
            },
            description="Apply gamma, log, or sigmoid tone correction.",
            subcategory="Contrast",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(params, (("input_max", "input-max"),))
        self.mode = str(normalized.get("mode", "gamma"))
        self.gamma = float(normalized.get("gamma", 1.0))
        self.gain = float(normalized.get("gain", 1.0))
        self.cutoff = float(normalized.get("cutoff", 0.5))
        self.inverse = parse_bool(normalized.get("inverse", False))
        self.input_max = (
            float(normalized["input-max"]) if normalized.get("input-max") else None
        )
        if self.mode not in _MODES:
            raise ValueError("tone-curve mode must be gamma, log, or sigmoid")
        if self.gamma <= 0:
            raise ValueError("tone-curve gamma must be positive")
        if self.gain <= 0:
            raise ValueError("tone-curve gain must be positive")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, self.type_name)
        frame = normalized_float(packet.data, self.input_max)
        if self.mode == "gamma":
            adjusted = np.power(frame, self.gamma)
        elif self.mode == "log":
            adjusted = np.log1p(frame * self.gain) / np.log1p(self.gain)
        else:
            sign = -1.0 if self.inverse else 1.0
            adjusted = 1.0 / (1.0 + np.exp(sign * self.gain * (self.cutoff - frame)))
        if self.mode != "sigmoid" and self.inverse:
            adjusted = 1.0 - adjusted
        output = from_normalized_float(np.clip(adjusted, 0.0, 1.0), packet.data.dtype)
        metadata = enhanced_metadata(
            packet,
            output,
            self.instance_id,
            self.type_name,
            {
                "mode": self.mode,
                "gamma": self.gamma,
                "gain": self.gain,
                "cutoff": self.cutoff,
                "inverse": self.inverse,
                "input-max": self.input_max,
            },
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}
