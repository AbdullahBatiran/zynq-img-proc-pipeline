"""Laplacian-of-Gaussian transformer."""

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
    parse_bool,
    validate_filter_packet,
    validate_odd_kernel_size,
)


_MODES = {"abs", "signed", "sharpen"}


class LogFilter(Transformer):
    """Apply Laplacian-of-Gaussian edge/blob enhancement."""

    type_name = "log-filter"

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
                "sigma": ParameterContract("sigma", "float", default=1.0),
                "kernel-size": ParameterContract("kernel-size", "int", default=0),
                "amount": ParameterContract("amount", "float", default=1.0),
                "mode": ParameterContract("mode", "str", default="abs", choices=tuple(sorted(_MODES))),
                "normalize": ParameterContract("normalize", "bool", default=True),
            },
            description="Enhance blobs and edges with a Laplacian-of-Gaussian filter.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(params, (("kernel_size", "kernel-size"),))
        self.sigma = float(normalized.get("sigma", 1.0))
        self.kernel_size = int(normalized.get("kernel-size", 0))
        self.amount = float(normalized.get("amount", 1.0))
        self.mode = str(normalized.get("mode", "abs"))
        self.normalize = parse_bool(normalized.get("normalize", True))
        if self.sigma <= 0:
            raise ValueError("log-filter sigma must be positive")
        validate_odd_kernel_size(self.kernel_size, "log-filter kernel-size", allow_zero=True)
        if self.mode not in _MODES:
            raise ValueError("log-filter mode must be abs, signed, or sharpen")

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
                "sigma": self.sigma,
                "kernel-size": self.kernel_size,
                "amount": self.amount,
                "mode": self.mode,
                "normalize": self.normalize,
            },
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}

    def _plane(self, plane: np.ndarray) -> np.ndarray:
        working = plane.astype(np.float64)
        blurred = cv2.GaussianBlur(working, (0, 0), self.sigma)
        ksize = self.kernel_size or 3
        lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
        if self.mode == "sharpen":
            result = working - self.amount * lap
            return clip_cast_preserve_dtype(result, plane.dtype)
        result = np.abs(lap) if self.mode == "abs" else lap
        result *= self.amount
        if self.normalize:
            return minmax_normalize(result, plane.dtype)
        return clip_cast_preserve_dtype(result, plane.dtype)
