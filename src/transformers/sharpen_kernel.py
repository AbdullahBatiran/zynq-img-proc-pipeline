"""Fixed sharpen kernel transformer."""

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
)


_KERNELS = {
    "cross": np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 5.0, -1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    ),
    "full": np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, 9.0, -1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    ),
}


class SharpenKernel(Transformer):
    """Sharpen frames with fixed 3x3 convolution kernels."""

    type_name = "sharpen-kernel"

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
                "kernel": ParameterContract(
                    "kernel",
                    "str",
                    default="cross",
                    choices=tuple(sorted(_KERNELS)),
                    description="Fixed sharpen kernel.",
                ),
                "iterations": ParameterContract(
                    "iterations",
                    "int",
                    default=1,
                    description="Number of times to apply the kernel.",
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
            description="Sharpen frames with fixed 3x3 convolution kernels.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(
            params,
            (("range_mode", "range-mode"), ("output_bits", "output-bits")),
        )
        self.kernel_name = str(normalized.get("kernel", "cross"))
        if self.kernel_name not in _KERNELS:
            raise ValueError("sharpen-kernel kernel must be cross or full")
        self.iterations = int(normalized.get("iterations", 1))
        self.range_mode = str(normalized.get("range-mode", "clip"))
        self.output_bits = (
            int(normalized["output-bits"])
            if normalized.get("output-bits") is not None
            else None
        )
        if self.iterations <= 0:
            raise ValueError("sharpen-kernel iterations must be positive")
        if self.range_mode not in RANGE_MODES:
            raise ValueError("sharpen-kernel range-mode must be 'clip' or 'limit'")
        if self.output_bits is not None and self.output_bits <= 0:
            raise ValueError("sharpen-kernel output-bits must be a positive integer")
        if self.output_bits is not None and self.output_bits > 16:
            raise ValueError("sharpen-kernel output-bits cannot exceed 16")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, self.type_name)
        resolved_bits, output_max = resolve_output_bits_and_max(
            packet.data.dtype, self.output_bits, self.type_name
        )

        sharpened = packet.data.astype(np.float64)
        kernel = _KERNELS[self.kernel_name]
        for _ in range(self.iterations):
            base = sharpened
            candidate = cv2.filter2D(base, cv2.CV_64F, kernel)
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
                "kernel": self.kernel_name,
                "iterations": self.iterations,
                "range-mode": self.range_mode,
                "output-bits": resolved_bits,
            },
        )
        return {"out": [FramePacket(data=sharpened, metadata=metadata)]}
