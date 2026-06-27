"""Total variation denoising transformer."""

from __future__ import annotations

from typing import Any

from skimage.restoration import denoise_tv_chambolle

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket
from src.transformers._filter_utils import (
    apply_per_channel,
    enhanced_metadata,
    from_normalized_float,
    normalize_aliases,
    normalized_float,
    validate_filter_packet,
)


class TvDenoise(Transformer):
    """Apply total variation denoising."""

    type_name = "tv-denoise"

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
                "weight": ParameterContract("weight", "float", default=0.1),
                "eps": ParameterContract("eps", "float", default=0.0002),
                "max-num-iter": ParameterContract("max-num-iter", "int", default=200),
            },
            description="Apply total variation denoising.",
            subcategory="Filter",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized = normalize_aliases(params, (("max_num_iter", "max-num-iter"),))
        self.weight = float(normalized.get("weight", 0.1))
        self.eps = float(normalized.get("eps", 0.0002))
        self.max_num_iter = int(normalized.get("max-num-iter", 200))
        if self.weight <= 0:
            raise ValueError("tv-denoise weight must be positive")
        if self.eps <= 0:
            raise ValueError("tv-denoise eps must be positive")
        if self.max_num_iter <= 0:
            raise ValueError("tv-denoise max-num-iter must be positive")

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
                "weight": self.weight,
                "eps": self.eps,
                "max-num-iter": self.max_num_iter,
            },
        )
        return {"out": [FramePacket(data=output, metadata=metadata)]}

    def _plane(self, plane):
        denoised = denoise_tv_chambolle(
            normalized_float(plane),
            weight=self.weight,
            eps=self.eps,
            max_num_iter=self.max_num_iter,
            channel_axis=None,
        )
        return from_normalized_float(denoised, plane.dtype)
