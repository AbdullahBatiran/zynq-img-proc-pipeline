"""Contrast histogram equalization transformer."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.lib.contracts import ElementContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket, infer_frame_shape


class HistEqualize(Transformer):
    """Apply global histogram equalization or CLAHE."""

    type_name = "hist_equalize"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={
                "in": PortContract(
                    "in", formats={"bgr", "rgb", "gray"}, depths={8}
                )
            },
            output_ports={"out": PortContract("out", depths={8})},
            description="Apply contrast histogram equalization.",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.mode = str(params.get("mode", "global"))
        if self.mode not in {"global", "clahe"}:
            raise ValueError("hist_equalize mode must be 'global' or 'clahe'")
        self.clip_limit = float(params.get("clip_limit", 2.0))
        self.tile_grid_size = _parse_tile_grid_size(params.get("tile_grid_size", "8x8"))

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        if packet.metadata.depth != 8:
            raise ValueError("hist_equalize currently supports 8-bit frames only")
        if packet.metadata.format not in {"bgr", "rgb", "gray"}:
            raise ValueError(
                f"hist_equalize does not support format {packet.metadata.format!r}"
            )

        equalized = self._equalize(packet.data, packet.metadata.format)
        width, height, channels, depth = infer_frame_shape(equalized)
        metadata = packet.metadata.derive(
            width=width,
            height=height,
            channels=channels,
            depth=depth,
            extra={**packet.metadata.extra, "hist_equalized_by": self.instance_id},
        )
        return {"out": [FramePacket(data=equalized, metadata=metadata)]}

    def _equalize(self, frame: np.ndarray, frame_format: str) -> np.ndarray:
        if frame_format == "gray":
            return self._equalize_plane(frame)

        if frame_format == "bgr":
            to_ycrcb = cv2.COLOR_BGR2YCrCb
            from_ycrcb = cv2.COLOR_YCrCb2BGR
        else:
            to_ycrcb = cv2.COLOR_RGB2YCrCb
            from_ycrcb = cv2.COLOR_YCrCb2RGB

        ycrcb = cv2.cvtColor(frame, to_ycrcb)
        ycrcb[:, :, 0] = self._equalize_plane(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, from_ycrcb)

    def _equalize_plane(self, plane: np.ndarray) -> np.ndarray:
        if self.mode == "global":
            return cv2.equalizeHist(plane)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        return clahe.apply(plane)


def _parse_tile_grid_size(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str) and "x" in value:
        left, right = value.lower().split("x", 1)
        return int(left), int(right)
    raise ValueError("tile_grid_size must be a tuple/list pair or string like '8x8'")
