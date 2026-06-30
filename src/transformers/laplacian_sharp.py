"""Laplacian sharpening transformer."""

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


_MODES = {"subtract", "add"}


class LaplacianSharp(Transformer):
    """Sharpen frames by repeatedly applying a Laplacian contribution."""

    type_name = "laplacian-sharp"

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
                    description="Laplacian contribution strength.",
                ),
                "kernel-size": ParameterContract(
                    "kernel-size",
                    "int",
                    default=3,
                    description="Laplacian aperture size.",
                ),
                "iterations": ParameterContract(
                    "iterations",
                    "int",
                    default=1,
                    description="Number of Laplacian contributions to apply.",
                ),
                "mode": ParameterContract(
                    "mode",
                    "str",
                    default="subtract",
                    choices=tuple(sorted(_MODES)),
                    description="Subtract or add the Laplacian contribution.",
                ),
                "scale": ParameterContract(
                    "scale",
                    "float",
                    default=1.0,
                    description="Scale passed to cv2.Laplacian.",
                ),
                "delta": ParameterContract(
                    "delta",
                    "float",
                    default=0.0,
                    description="Delta passed to cv2.Laplacian.",
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
            description="Sharpen frames with one or more Laplacian passes.",
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
        self.kernel_size = int(normalized.get("kernel-size", 3))
        self.iterations = int(normalized.get("iterations", 1))
        self.mode = str(normalized.get("mode", "subtract"))
        self.scale = float(normalized.get("scale", 1.0))
        self.delta = float(normalized.get("delta", 0.0))
        self.range_mode = str(normalized.get("range-mode", "clip"))
        self.output_bits = (
            int(normalized["output-bits"])
            if normalized.get("output-bits") is not None
            else None
        )

        if self.amount < 0:
            raise ValueError("laplacian-sharp amount must be non-negative")
        validate_odd_kernel_size(self.kernel_size, "laplacian-sharp kernel-size")
        if self.iterations <= 0:
            raise ValueError("laplacian-sharp iterations must be positive")
        if self.mode not in _MODES:
            raise ValueError("laplacian-sharp mode must be 'subtract' or 'add'")
        if self.scale < 0:
            raise ValueError("laplacian-sharp scale must be non-negative")
        if self.range_mode not in RANGE_MODES:
            raise ValueError("laplacian-sharp range-mode must be 'clip' or 'limit'")
        if self.output_bits is not None and self.output_bits <= 0:
            raise ValueError("laplacian-sharp output-bits must be a positive integer")
        if self.output_bits is not None and self.output_bits > 16:
            raise ValueError("laplacian-sharp output-bits cannot exceed 16")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, "laplacian-sharp")
        resolved_bits, output_max = resolve_output_bits_and_max(
            packet.data.dtype, self.output_bits, "laplacian-sharp"
        )
        sharpened = packet.data.astype(np.float64)
        for _ in range(self.iterations):
            base = sharpened
            laplacian = cv2.Laplacian(
                base,
                cv2.CV_64F,
                ksize=self.kernel_size,
                scale=self.scale,
                delta=self.delta,
            )
            if self.mode == "subtract":
                candidate = base - self.amount * laplacian
            else:
                candidate = base + self.amount * laplacian
            if self.range_mode == "limit":
                sharpened = limit_sharpened_detail(base, candidate, output_max)
            else:
                sharpened = candidate

        sharpened = clip_cast_to_output_max(sharpened, packet.data.dtype, output_max)
        metadata = filtered_metadata(
            packet,
            sharpened,
            self.instance_id,
            self.type_name,
            {
                "amount": self.amount,
                "kernel-size": self.kernel_size,
                "iterations": self.iterations,
                "mode": self.mode,
                "scale": self.scale,
                "delta": self.delta,
                "range-mode": self.range_mode,
                "output-bits": resolved_bits,
            },
        )
        return {"out": [FramePacket(data=sharpened, metadata=metadata)]}
