"""Multi-input stream combiner."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.lib.contracts import ElementContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FrameMetadata, FramePacket, infer_frame_shape, new_packet_id


class Combine(Transformer):
    """Combine two synchronized streams into one output stream."""

    type_name = "combine"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"left": PortContract("left"), "right": PortContract("right")},
            output_ports={"out": PortContract("out")},
            description="Combine two streams horizontally, vertically, or by overlay.",
            require_same_format=True,
            require_same_depth=True,
            require_same_index=True,
            synchronized_inputs=True,
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.mode = str(params.get("mode", "horizontal"))
        if self.mode not in {"horizontal", "vertical", "overlay"}:
            raise ValueError("combine mode must be horizontal, vertical, or overlay")
        self.alpha = float(params.get("alpha", 0.5))
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("combine alpha must be between 0.0 and 1.0")
        self.stream_id = params.get("stream_id")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        left = self._single_input(inputs, "left")
        right = self._single_input(inputs, "right")
        self._validate_pair(left, right)

        if self.mode == "horizontal":
            combined = np.hstack([left.data, right.data])
        elif self.mode == "vertical":
            combined = np.vstack([left.data, right.data])
        else:
            combined = cv2.addWeighted(left.data, self.alpha, right.data, 1 - self.alpha, 0)

        width, height, channels, depth = infer_frame_shape(combined)
        parents = tuple(
            dict.fromkeys(
                (
                    *left.metadata.parents,
                    left.metadata.packet_id,
                    *right.metadata.parents,
                    right.metadata.packet_id,
                )
            )
        )
        metadata = FrameMetadata(
            packet_id=new_packet_id(),
            stream_id=str(
                self.stream_id
                or f"{left.metadata.stream_id}+{right.metadata.stream_id}"
            ),
            source_id=f"{left.metadata.source_id}+{right.metadata.source_id}",
            pts=max(left.metadata.pts, right.metadata.pts),
            index=left.metadata.index,
            format=left.metadata.format,
            width=width,
            height=height,
            fps=min(left.metadata.fps, right.metadata.fps),
            depth=depth,
            channels=channels,
            parents=parents,
            extra={
                "combined_by": self.instance_id,
                "combine_mode": self.mode,
                "left_stream_id": left.metadata.stream_id,
                "right_stream_id": right.metadata.stream_id,
            },
        )
        return {"out": [FramePacket(data=combined, metadata=metadata)]}

    def _validate_pair(self, left: FramePacket, right: FramePacket) -> None:
        lm = left.metadata
        rm = right.metadata
        if lm.format != rm.format:
            raise ValueError("combine requires matching frame formats")
        if lm.depth != rm.depth:
            raise ValueError("combine requires matching frame depths")
        if lm.channels != rm.channels:
            raise ValueError("combine requires matching channel counts")
        if lm.index != rm.index:
            raise ValueError("combine requires matching frame indexes")

        if self.mode == "horizontal" and lm.height != rm.height:
            raise ValueError("horizontal combine requires equal heights")
        if self.mode == "vertical" and lm.width != rm.width:
            raise ValueError("vertical combine requires equal widths")
        if self.mode == "overlay" and (
            lm.width != rm.width or lm.height != rm.height
        ):
            raise ValueError("overlay combine requires equal dimensions")
