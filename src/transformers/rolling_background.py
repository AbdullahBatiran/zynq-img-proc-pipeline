"""Rolling-ball background correction transformer."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.restoration import rolling_ball

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    apply_per_channel,
    clip_cast_preserve_dtype,
    enhanced_metadata,
    minmax_normalize,
    parse_bool,
    validate_filter_packet,
)


_MODES = {"subtract", "divide"}


class RollingBackground(Transformer):
    """Correct uneven background using rolling-ball estimation."""

    type_name = "rolling-background"

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
                "radius": ParameterContract("radius", "int", default=50),
                "mode": ParameterContract(
                    "mode", "str", default="subtract", choices=tuple(sorted(_MODES))
                ),
                "normalize": ParameterContract("normalize", "bool", default=False),
            },
            description="Correct rolling background gradients.",
            subcategory="Contrast",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.radius = int(params.get("radius", 50))
        self.mode = str(params.get("mode", "subtract"))
        self.normalize = parse_bool(params.get("normalize", False))
        if self.radius <= 0:
            raise ValueError("rolling-background radius must be positive")
        if self.mode not in _MODES:
            raise ValueError("rolling-background mode must be subtract or divide")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, self.type_name)
        output = apply_per_channel(packet.data, self._plane)
        metadata = enhanced_metadata(
            packet,
            output,
            self.instance_id,
            self.type_name,
            {"radius": self.radius, "mode": self.mode, "normalize": self.normalize},
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}

    def _plane(self, plane: np.ndarray) -> np.ndarray:
        working = plane.astype(np.float64)
        background = rolling_ball(working, radius=self.radius)
        if self.mode == "subtract":
            result = working - background
        else:
            result = working * (float(np.mean(background)) + 1e-6) / (
                background + 1e-6
            )
        if self.normalize:
            return minmax_normalize(result, plane.dtype)
        return clip_cast_preserve_dtype(result, plane.dtype)
