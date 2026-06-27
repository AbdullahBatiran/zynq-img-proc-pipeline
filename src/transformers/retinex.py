"""Retinex-style local contrast transformer."""

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
    minmax_normalize,
    normalize_aliases,
    parse_float_list,
    validate_filter_packet,
)


_MODES = {"single", "multi"}
_OUTPUT_MODES = {"preserve", "normalize"}


class Retinex(Transformer):
    """Apply single-scale or multi-scale Retinex local contrast."""

    type_name = "retinex"

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
                    "mode", "str", default="multi", choices=tuple(sorted(_MODES))
                ),
                "sigma": ParameterContract("sigma", "float", default=15.0),
                "sigmas": ParameterContract("sigmas", "list", default="15,80,250"),
                "gain": ParameterContract("gain", "float", default=1.0),
                "offset": ParameterContract("offset", "float", default=0.0),
                "output-mode": ParameterContract(
                    "output-mode",
                    "str",
                    default="preserve",
                    choices=tuple(sorted(_OUTPUT_MODES)),
                ),
            },
            description="Apply Retinex-style local contrast enhancement.",
            subcategory="Contrast",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(params, (("output_mode", "output-mode"),))
        self.mode = str(normalized.get("mode", "multi"))
        self.sigma = float(normalized.get("sigma", 15.0))
        self.sigmas = parse_float_list(normalized.get("sigmas", "15,80,250"))
        self.gain = float(normalized.get("gain", 1.0))
        self.offset = float(normalized.get("offset", 0.0))
        self.output_mode = str(normalized.get("output-mode", "preserve"))
        if self.mode not in _MODES:
            raise ValueError("retinex mode must be single or multi")
        if self.output_mode not in _OUTPUT_MODES:
            raise ValueError("retinex output-mode must be preserve or normalize")
        if self.sigma <= 0 or any(sigma <= 0 for sigma in self.sigmas):
            raise ValueError("retinex sigma values must be positive")

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
                "mode": self.mode,
                "sigma": self.sigma,
                "sigmas": self.sigmas,
                "gain": self.gain,
                "offset": self.offset,
                "output-mode": self.output_mode,
            },
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}

    def _plane(self, plane: np.ndarray) -> np.ndarray:
        sigmas = [self.sigma] if self.mode == "single" else self.sigmas
        working = plane.astype(np.float64) + 1.0
        retinex = np.zeros_like(working)
        for sigma in sigmas:
            blur = cv2.GaussianBlur(working, (0, 0), sigma) + 1.0
            retinex += np.log(working) - np.log(blur)
        retinex /= len(sigmas)
        if self.output_mode == "normalize":
            return minmax_normalize(retinex, plane.dtype)
        result = plane.astype(np.float64) + self.gain * retinex + self.offset
        return clip_cast_preserve_dtype(result, plane.dtype)
