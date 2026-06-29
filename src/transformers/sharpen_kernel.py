"""Fixed sharpen kernel transformer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    clip_cast_preserve_dtype,
    filtered_metadata,
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
            },
            description="Sharpen frames with fixed 3x3 convolution kernels.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.kernel_name = str(params.get("kernel", "cross"))
        if self.kernel_name not in _KERNELS:
            raise ValueError("sharpen-kernel kernel must be cross or full")
        self.iterations = int(params.get("iterations", 1))
        if self.iterations <= 0:
            raise ValueError("sharpen-kernel iterations must be positive")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        validate_filter_packet(packet, self.type_name)

        sharpened = packet.data.astype(np.float64)
        kernel = _KERNELS[self.kernel_name]
        for _ in range(self.iterations):
            sharpened = cv2.filter2D(sharpened, cv2.CV_64F, kernel)
        sharpened = clip_cast_preserve_dtype(sharpened, packet.data.dtype)

        metadata = filtered_metadata(
            packet,
            sharpened,
            self.instance_id,
            self.type_name,
            {
                "kernel": self.kernel_name,
                "iterations": self.iterations,
            },
        )
        return {"out": [FramePacket(data=sharpened, metadata=metadata)]}
