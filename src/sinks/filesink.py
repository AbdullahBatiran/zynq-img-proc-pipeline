"""Video file sink element."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, PipelineContext, Sink


class FileSink(Sink):
    """Write frames to a video file."""

    type_name = "filesink"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in")},
            parameters={
                "path": ParameterContract(
                    "path", "path", required=True, description="Output video path."
                ),
                "codec": ParameterContract(
                    "codec",
                    "str",
                    default="mp4v",
                    description="FourCC codec used by OpenCV VideoWriter.",
                ),
                "fps": ParameterContract(
                    "fps",
                    "float",
                    default="<input metadata fps or 30>",
                    description="Output FPS override.",
                ),
            },
            description="Write video frames to a file with OpenCV.",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.path = Path(str(params["path"]))
        self.codec = str(params.get("codec", "mp4v"))
        self.fps = params.get("fps")
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None
        self.is_color: bool | None = None

    def start(self, context: PipelineContext) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        metadata = packet.metadata
        frame_size = (metadata.width, metadata.height)
        is_color = metadata.channels != 1
        if self.writer is None:
            fps = float(self.fps or metadata.fps or 30.0)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, frame_size, is_color)
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {self.path}")
            self.size = frame_size
            self.is_color = is_color
        elif self.size != frame_size:
            raise ValueError("filesink received changing frame dimensions")

        self.writer.write(packet.data)
        return {}

    def stop(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
