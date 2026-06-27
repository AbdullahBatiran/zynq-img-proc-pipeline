"""Display sink element."""

from __future__ import annotations

from typing import Any

import cv2

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, PipelineContext, Sink


class DisplaySink(Sink):
    """Display frames in an OpenCV window."""

    type_name = "displaysink"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in")},
            parameters={
                "window_name": ParameterContract(
                    "window_name",
                    "str",
                    default="<element id>",
                    description="OpenCV display window name.",
                ),
                "wait_ms": ParameterContract(
                    "wait_ms",
                    "int",
                    default="<derived from fps when sync=true>",
                    description="Fixed cv2.waitKey delay in milliseconds.",
                ),
                "fps": ParameterContract(
                    "fps",
                    "float",
                    default="<input metadata fps>",
                    description="Playback FPS override when sync is enabled.",
                ),
                "sync": ParameterContract(
                    "sync",
                    "bool",
                    default=True,
                    description="Use FPS to pace display instead of running as fast as possible.",
                ),
                "enabled": ParameterContract(
                    "enabled",
                    "bool",
                    default=True,
                    description="Disable OpenCV display calls for tests/headless runs.",
                ),
                "quit_key": ParameterContract(
                    "quit_key",
                    "str",
                    default="q",
                    description="Keyboard key that requests pipeline stop.",
                ),
            },
            description="Display video frames in an OpenCV window.",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.window_name = str(params.get("window_name", self.instance_id))
        self.wait_ms = (
            int(params["wait_ms"]) if params.get("wait_ms") is not None else None
        )
        self.fps = float(params["fps"]) if params.get("fps") is not None else None
        self.sync = bool(params.get("sync", True))
        self.enabled = bool(params.get("enabled", True))
        self.quit_key = str(params.get("quit_key", "q"))
        self.context: PipelineContext | None = None

    def start(self, context: PipelineContext) -> None:
        self.context = context

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        if self.enabled:
            cv2.imshow(self.window_name, packet.data)
            key = cv2.waitKey(self._wait_ms(packet)) & 0xFF
            if self.quit_key and key == ord(self.quit_key[0]):
                if self.context is not None:
                    self.context.request_stop()
        return {}

    def stop(self) -> None:
        if self.enabled:
            cv2.destroyWindow(self.window_name)
        self.context = None

    def _wait_ms(self, packet) -> int:
        if self.wait_ms is not None:
            return max(1, self.wait_ms)
        if self.sync:
            fps = self.fps if self.fps is not None else packet.metadata.fps
            if fps and fps > 0:
                return max(1, round(1000.0 / fps))
        return 1
