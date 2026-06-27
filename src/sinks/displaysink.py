"""Display sink element."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

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
                "autorange": ParameterContract(
                    "autorange",
                    "bool",
                    default=False,
                    description="Stretch each frame to the dtype display range.",
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
        self.autorange = bool(params.get("autorange", False))
        self.quit_key = str(params.get("quit_key", "q"))
        self.context: PipelineContext | None = None

    def start(self, context: PipelineContext) -> None:
        self.context = context

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        if self.enabled:
            cv2.imshow(self.window_name, self._display_frame(packet.data))
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

    def _display_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.autorange:
            return frame
        if frame.size == 0:
            return frame

        input_min = float(np.min(frame))
        input_max = float(np.max(frame))
        if input_min >= input_max:
            return np.zeros_like(frame)

        output_min, output_max = _display_range_for_dtype(frame.dtype)
        scaled = (
            (frame.astype(np.float64) - input_min)
            * (output_max - output_min)
            / (input_max - input_min)
            + output_min
        )
        scaled = np.clip(scaled, output_min, output_max)
        if np.issubdtype(frame.dtype, np.integer):
            scaled = np.rint(scaled)
        return scaled.astype(frame.dtype)


def _display_range_for_dtype(dtype: np.dtype) -> tuple[float, float]:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    return 0.0, 1.0
