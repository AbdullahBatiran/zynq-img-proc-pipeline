"""Progress reporting pass-through transformer."""

from __future__ import annotations

import sys
import time
from typing import Any, TextIO

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, PipelineContext, Transformer


_STREAMS = {"stdout", "stderr", "file"}


class Progress(Transformer):
    """Print progress records while passing packets through unchanged."""

    type_name = "progress"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in")},
            output_ports={"out": PortContract("out")},
            parameters={
                "every-frames": ParameterContract("every-frames", "int", default=30),
                "every-seconds": ParameterContract("every-seconds", "float", default=None),
                "stream": ParameterContract("stream", "str", default="stdout", choices=tuple(sorted(_STREAMS))),
                "path": ParameterContract("path", "path", default=None),
                "label": ParameterContract("label", "str", default="<element id>"),
            },
            description="Print stream progress while passing packets through.",
            subcategory="Debug",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.every_frames = int(params.get("every-frames", 30))
        self.every_seconds = (
            float(params["every-seconds"]) if params.get("every-seconds") else None
        )
        self.stream = str(params.get("stream", "stdout"))
        self.path = str(params["path"]) if params.get("path") else None
        self.label = str(params.get("label", self.instance_id))
        if self.every_frames <= 0:
            raise ValueError("progress every-frames must be positive")
        if self.every_seconds is not None and self.every_seconds <= 0:
            raise ValueError("progress every-seconds must be positive")
        if self.stream not in _STREAMS:
            raise ValueError("progress stream must be stdout, stderr, or file")
        if self.stream == "file" and not self.path:
            raise ValueError("progress path is required when stream=file")
        self.frames_seen = 0
        self.start_time = 0.0
        self.last_pts: float | None = None
        self.file: TextIO | None = None

    def start(self, context: PipelineContext) -> None:
        self.start_time = time.monotonic()

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        if self._should_print(packet.metadata.pts):
            elapsed = max(time.monotonic() - self.start_time, 1e-9)
            fps = (self.frames_seen + 1) / elapsed
            print(
                (
                    f"[progress {self.label}] frames={self.frames_seen + 1} "
                    f"index={packet.metadata.index} pts={packet.metadata.pts:.6f} "
                    f"stream={packet.metadata.stream_id} elapsed={elapsed:.3f}s "
                    f"fps={fps:.3f}"
                ),
                file=self._output_stream(),
            )
            self._output_stream().flush()
            self.last_pts = packet.metadata.pts
        self.frames_seen += 1
        return {"out": [packet]}

    def stop(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def _should_print(self, pts: float) -> bool:
        if self.every_seconds is not None:
            return self.last_pts is None or pts - self.last_pts >= self.every_seconds
        return self.frames_seen % self.every_frames == 0

    def _output_stream(self) -> TextIO:
        if self.stream == "stdout":
            return sys.stdout
        if self.stream == "stderr":
            return sys.stderr
        if self.file is None:
            self.file = open(self.path or "", "a", encoding="utf-8")
        return self.file
