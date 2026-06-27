"""Video file source element."""

from __future__ import annotations

from typing import Any

import cv2

from src.lib.contracts import ElementContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, PipelineContext, Source
from src.lib.packets import FrameMetadata, FramePacket, infer_frame_shape, new_packet_id


class FileSource(Source):
    """Read video frames from a file."""

    type_name = "filesrc"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            output_ports={"out": PortContract("out")},
            description="Read video frames from a file with OpenCV.",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.path = str(params["path"])
        self.stream_id = str(params.get("stream_id", self.instance_id))
        self.source_id = str(params.get("source_id", self.instance_id))
        self.format = str(params.get("format", "bgr"))
        self.cap: cv2.VideoCapture | None = None
        self.index = 0
        self.fps = 0.0

    def start(self, context: PipelineContext) -> None:
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file {self.path!r}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        if self.cap is None:
            raise RuntimeError("filesrc was not started")
        ok, frame = self.cap.read()
        if not ok:
            return {}

        width, height, channels, depth = infer_frame_shape(frame)
        pts_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        metadata = FrameMetadata(
            packet_id=new_packet_id(),
            stream_id=self.stream_id,
            source_id=self.source_id,
            pts=pts_ms / 1000.0,
            index=self.index,
            format=self.format,
            width=width,
            height=height,
            fps=self.fps,
            depth=depth,
            channels=channels,
            parents=(),
            extra={"path": self.path},
        )
        self.index += 1
        return {"out": [FramePacket(data=frame, metadata=metadata)]}

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
