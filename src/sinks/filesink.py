"""Video file sink element."""

from __future__ import annotations

from pathlib import Path
import sys
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
                "quality": ParameterContract(
                    "quality",
                    "int",
                    default="<backend default>",
                    description="Encoder quality hint from 0 to 100 when supported.",
                ),
            },
            description="Write video frames to a file with OpenCV.",
            subcategory="File",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.path = Path(str(params["path"]))
        self.codec = str(params.get("codec", "mp4v"))
        self.fps = params.get("fps")
        self.quality = (
            int(params["quality"]) if params.get("quality") is not None else None
        )
        if self.quality is not None and not 0 <= self.quality <= 100:
            raise ValueError("filesink quality must be in the range 0..100")
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
            if self.quality is not None and not self.writer.set(
                cv2.VIDEOWRITER_PROP_QUALITY, float(self.quality)
            ):
                print(
                    "Warning: filesink quality option was not accepted by the "
                    f"OpenCV backend for codec {self.codec!r}",
                    file=sys.stderr,
                )
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
