"""Wavelet denoising transformer."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    apply_per_channel,
    enhanced_metadata,
    from_normalized_float,
    normalized_float,
    parse_bool,
    validate_filter_packet,
)


_MODES = {"soft", "hard"}


class WaveletDenoise(Transformer):
    """Apply wavelet denoising."""

    type_name = "wavelet-denoise"

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
                "sigma": ParameterContract("sigma", "float|str", default="auto"),
                "wavelet": ParameterContract("wavelet", "str", default="db1"),
                "mode": ParameterContract("mode", "str", default="soft", choices=tuple(sorted(_MODES))),
                "rescale-sigma": ParameterContract("rescale-sigma", "bool", default=True),
            },
            description="Apply wavelet denoising.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        sigma = params.get("sigma", "auto")
        self.sigma = None if str(sigma).lower() == "auto" else float(sigma)
        self.wavelet = str(params.get("wavelet", "db1"))
        self.mode = str(params.get("mode", "soft"))
        self.rescale_sigma = parse_bool(params.get("rescale-sigma", True))
        if self.sigma is not None and self.sigma < 0:
            raise ValueError("wavelet-denoise sigma must be non-negative")
        if self.mode not in _MODES:
            raise ValueError("wavelet-denoise mode must be soft or hard")
        if self.wavelet not in {"db1", "haar"}:
            raise ValueError("wavelet-denoise supports db1/haar without extra deps")

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
                "sigma": self.sigma if self.sigma is not None else "auto",
                "wavelet": self.wavelet,
                "mode": self.mode,
                "rescale-sigma": self.rescale_sigma,
            },
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}

    def _plane(self, plane):
        denoised = _haar_denoise(normalized_float(plane), self.sigma, self.mode)
        return from_normalized_float(denoised, plane.dtype)


def _threshold(values: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    if mode == "hard":
        return values * (np.abs(values) >= threshold)
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _haar_denoise(frame: np.ndarray, sigma: float | None, mode: str) -> np.ndarray:
    height, width = frame.shape
    padded_height = height + height % 2
    padded_width = width + width % 2
    padded = np.pad(
        frame,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="edge",
    )
    a = padded[0::2, 0::2]
    b = padded[0::2, 1::2]
    c = padded[1::2, 0::2]
    d = padded[1::2, 1::2]
    ll = (a + b + c + d) / 2.0
    lh = (a - b + c - d) / 2.0
    hl = (a + b - c - d) / 2.0
    hh = (a - b - c + d) / 2.0
    if sigma is None:
        sigma = float(np.median(np.abs(hh - np.median(hh))) / 0.6745)
    threshold = sigma * np.sqrt(2.0 * np.log(max(frame.size, 2)))
    lh = _threshold(lh, threshold, mode)
    hl = _threshold(hl, threshold, mode)
    hh = _threshold(hh, threshold, mode)
    reconstructed = np.empty_like(padded)
    reconstructed[0::2, 0::2] = (ll + lh + hl + hh) / 2.0
    reconstructed[0::2, 1::2] = (ll - lh + hl - hh) / 2.0
    reconstructed[1::2, 0::2] = (ll + lh - hl - hh) / 2.0
    reconstructed[1::2, 1::2] = (ll - lh - hl + hh) / 2.0
    return np.clip(reconstructed[:height, :width], 0.0, 1.0)
