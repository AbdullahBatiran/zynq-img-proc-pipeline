"""Frame-difference diagnostics pass-through transformer."""

from __future__ import annotations

import sys
from typing import Any, TextIO

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer


class FrameDiffDebug(Transformer):
    """Print frame-to-frame difference metrics."""

    type_name = "frame-diff-debug"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in")},
            output_ports={"out": PortContract("out")},
            parameters={
                "every-frames": ParameterContract("every-frames", "int", default=30),
                "show-mean-abs": ParameterContract("show-mean-abs", "bool", default=True),
                "show-max-abs": ParameterContract("show-max-abs", "bool", default=True),
                "show-rms": ParameterContract("show-rms", "bool", default=True),
            },
            description="Print frame-to-frame difference diagnostics.",
            subcategory="Debug",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.every_frames = int(params.get("every-frames", 30))
        self.show_mean_abs = _bool(params.get("show-mean-abs", True))
        self.show_max_abs = _bool(params.get("show-max-abs", True))
        self.show_rms = _bool(params.get("show-rms", True))
        if self.every_frames <= 0:
            raise ValueError("frame-diff-debug every-frames must be positive")
        self.previous: np.ndarray | None = None
        self.frames_seen = 0

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        if self.previous is not None and self.frames_seen % self.every_frames == 0:
            diff = packet.data.astype(np.float64) - self.previous.astype(np.float64)
            values = [f"[frame-diff-debug {self.instance_id}] index={packet.metadata.index}"]
            abs_diff = np.abs(diff)
            if self.show_mean_abs:
                values.append(f"mean_abs={float(np.mean(abs_diff)):.6g}")
            if self.show_max_abs:
                values.append(f"max_abs={float(np.max(abs_diff)):.6g}")
            if self.show_rms:
                values.append(f"rms={float(np.sqrt(np.mean(diff * diff))):.6g}")
            print(" ".join(values), file=sys.stdout)
            sys.stdout.flush()
        self.previous = packet.data.copy()
        self.frames_seen += 1
        return {"out": [packet]}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)
