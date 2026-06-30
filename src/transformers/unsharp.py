"""Unsharp mask sharpening transformer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    RANGE_MODES,
    clip_cast_to_output_max,
    filtered_metadata,
    limit_sharpened_detail,
    normalize_aliases,
    resolve_output_bits_and_max,
    validate_filter_packet,
    validate_odd_kernel_size,
)


class Unsharp(Transformer):
    """Sharpen frames using an unsharp mask."""

    type_name = "unsharp"

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
                "amount": ParameterContract(
                    "amount",
                    "float",
                    default=1.0,
                    description="Unsharp mask strength.",
                ),
                "sigma": ParameterContract(
                    "sigma",
                    "float",
                    default=1.0,
                    description="Gaussian blur sigma for the unsharp mask.",
                ),
                "kernel-size": ParameterContract(
                    "kernel-size",
                    "int",
                    default=0,
                    description="Gaussian kernel size; 0 derives it from sigma.",
                ),
                "range-mode": ParameterContract(
                    "range-mode",
                    "str",
                    default="clip",
                    choices=tuple(sorted(RANGE_MODES)),
                    description="Clip output or limit sharpening detail to range.",
                ),
                "output-bits": ParameterContract(
                    "output-bits",
                    "int",
                    default="<container depth>",
                    description="Effective output bit depth within the dtype container.",
                ),
            },
            description="Sharpen frames using an unsharp mask.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(
            params,
            (
                ("kernel_size", "kernel-size"),
                ("range_mode", "range-mode"),
                ("output_bits", "output-bits"),
            ),
        )
        self.amount = float(normalized.get("amount", 1.0))
        self.sigma = float(normalized.get("sigma", 1.0))
        self.kernel_size = int(normalized.get("kernel-size", 0))
        self.range_mode = str(normalized.get("range-mode", "clip"))
        self.output_bits = (
            int(normalized["output-bits"])
            if normalized.get("output-bits") is not None
            else None
        )

        if self.amount < 0:
            raise ValueError("Unsharp amount must be non-negative")
        if self.sigma <= 0:
            raise ValueError("Unsharp sigma must be positive")
        validate_odd_kernel_size(
            self.kernel_size, "Unsharp kernel-size", allow_zero=True
        )
        if self.range_mode not in RANGE_MODES:
            raise ValueError("Unsharp range-mode must be 'clip' or 'limit'")
        if self.output_bits is not None and self.output_bits <= 0:
            raise ValueError("Unsharp output-bits must be a positive integer")
        if self.output_bits is not None and self.output_bits > 16:
            raise ValueError("Unsharp output-bits cannot exceed 16")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, "Unsharp")
        resolved_bits, output_max = resolve_output_bits_and_max(
            packet.data.dtype, self.output_bits, "Unsharp"
        )
        ksize = (self.kernel_size, self.kernel_size)
        blurred = cv2.GaussianBlur(packet.data, ksize, self.sigma)
        base = packet.data.astype(np.float64)
        sharpened = (
            base * (1.0 + self.amount)
            - blurred.astype(np.float64) * self.amount
        )
        if self.range_mode == "limit":
            sharpened = limit_sharpened_detail(base, sharpened, output_max)
        sharpened = clip_cast_to_output_max(sharpened, packet.data.dtype, output_max)
        metadata = filtered_metadata(
            packet,
            sharpened,
            self.instance_id,
            self.type_name,
            {
                "amount": self.amount,
                "sigma": self.sigma,
                "kernel-size": self.kernel_size,
                "range-mode": self.range_mode,
                "output-bits": resolved_bits,
            },
        )
        return {"out": [FramePacket(data=sharpened, metadata=metadata)]}
