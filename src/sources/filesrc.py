"""Video file source element."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, PipelineContext, Source
from src.lib.packets import FrameMetadata, FramePacket, infer_frame_shape, new_packet_id

_FORMATS = {"auto", "gray", "bgr", "rgb"}
_DEPTHS = {"auto", 8, 16}


class FileSource(Source):
    """Read video frames from a file."""

    type_name = "filesrc"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            output_ports={"out": PortContract("out")},
            parameters={
                "path": ParameterContract(
                    "path", "path", required=True, description="Video file path."
                ),
                "stream_id": ParameterContract(
                    "stream_id",
                    "str",
                    default="<element id>",
                    description="Logical stream id assigned to emitted frames.",
                ),
                "source_id": ParameterContract(
                    "source_id",
                    "str",
                    default="<element id>",
                    description="Source id stored in frame metadata.",
                ),
                "format": ParameterContract(
                    "format",
                    "str",
                    default="bgr",
                    choices=("auto", "gray", "bgr", "rgb"),
                    description="Requested output pixel format.",
                ),
                "depth": ParameterContract(
                    "depth",
                    "int|str",
                    default="auto",
                    choices=("auto", 8, 16),
                    description="Requested decoded bit depth.",
                ),
                "preserve_native": ParameterContract(
                    "preserve_native",
                    "bool",
                    default=False,
                    description="Ask OpenCV not to convert decoded frames to RGB/BGR.",
                ),
                "strict": ParameterContract(
                    "strict",
                    "bool",
                    default="<true if format/depth requested, else false>",
                    description="Reject decoded frames that do not match requested format/depth.",
                ),
            },
            description="Read video frames from a file with OpenCV.",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.path = str(params["path"])
        self.stream_id = str(params.get("stream_id", self.instance_id))
        self.source_id = str(params.get("source_id", self.instance_id))
        self.format_was_requested = "format" in params
        self.depth_was_requested = "depth" in params
        self.format = str(params.get("format", "bgr")).lower()
        self.depth = _parse_depth(params.get("depth", "auto"))
        self.preserve_native = _parse_bool(params.get("preserve_native", False))
        self.strict = _parse_bool(
            params.get("strict", self.format_was_requested or self.depth_was_requested)
        )
        if self.format not in _FORMATS:
            raise ValueError(f"filesrc format must be one of {sorted(_FORMATS)}")
        if self.depth not in _DEPTHS:
            raise ValueError("filesrc depth must be auto, 8, or 16")
        self.cap: cv2.VideoCapture | None = None
        self.index = 0
        self.fps = 0.0

    def start(self, context: PipelineContext) -> None:
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file {self.path!r}")
        if self.preserve_native:
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        if self.cap is None:
            raise RuntimeError("filesrc was not started")
        ok, frame = self.cap.read()
        if not ok:
            return {}

        frame, actual_format = normalize_decoded_frame(
            frame=frame,
            requested_format=self.format,
            requested_depth=self.depth,
            strict=self.strict,
            path=self.path,
        )
        width, height, channels, depth = infer_frame_shape(frame)
        pts_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        metadata = FrameMetadata(
            packet_id=new_packet_id(),
            stream_id=self.stream_id,
            source_id=self.source_id,
            pts=pts_ms / 1000.0,
            index=self.index,
            format=actual_format,
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


def normalize_decoded_frame(
    *,
    frame: np.ndarray,
    requested_format: str,
    requested_depth: int | str,
    strict: bool,
    path: str,
) -> tuple[np.ndarray, str]:
    frame = _squeeze_single_channel(frame)
    actual_format = infer_frame_format(frame)
    actual_depth = frame.dtype.itemsize * 8
    requested_depth = _parse_depth(requested_depth)

    if requested_depth != "auto" and actual_depth != requested_depth:
        if strict:
            raise ValueError(
                f"filesrc expected {requested_depth}-bit frames from {path!r}, "
                f"but OpenCV decoded {actual_depth}-bit frames"
            )
        frame = frame.astype(_dtype_for_depth(requested_depth), copy=False)
        actual_depth = requested_depth

    if requested_format == "auto":
        return frame, actual_format

    if requested_format == actual_format:
        return frame, actual_format

    converted = _convert_format(frame, actual_format, requested_format)
    return converted, requested_format


def infer_frame_format(frame: np.ndarray) -> str:
    if frame.ndim == 2:
        return "gray"
    if frame.ndim == 3 and frame.shape[2] == 1:
        return "gray"
    if frame.ndim == 3 and frame.shape[2] == 3:
        return "bgr"
    raise ValueError(f"Unsupported decoded frame shape: {frame.shape}")


def _squeeze_single_channel(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    return frame


def _convert_format(
    frame: np.ndarray, actual_format: str, requested_format: str
) -> np.ndarray:
    if actual_format == "gray" and requested_format == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if actual_format == "gray" and requested_format == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if actual_format == "bgr" and requested_format == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if actual_format == "bgr" and requested_format == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if actual_format == "rgb" and requested_format == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if actual_format == "rgb" and requested_format == "bgr":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    raise ValueError(
        f"Cannot convert decoded frame from {actual_format!r} to {requested_format!r}"
    )


def _parse_depth(value: Any) -> int | str:
    if value == "auto":
        return "auto"
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    depth = int(value)
    if depth not in {8, 16}:
        raise ValueError("depth must be auto, 8, or 16")
    return depth


def _dtype_for_depth(depth: int | str) -> np.dtype:
    if depth == 8:
        return np.dtype(np.uint8)
    if depth == 16:
        return np.dtype(np.uint16)
    raise ValueError("Cannot infer dtype for auto depth")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)
