"""Debug pass-through transformer."""

from __future__ import annotations

import sys
from typing import Any, TextIO

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket


_STREAMS = {"stdout", "stderr", "file"}
_PREVIEW_MODES = {"top-left", "center", "both"}
_PARAMETER_NAMES = {
    "enabled",
    "every-seconds",
    "every-frames",
    "label",
    "stream",
    "path",
    "show-shape",
    "show-dtype",
    "show-min",
    "show-max",
    "show-mean",
    "show-std",
    "show-median",
    "show-preview",
    "show-percentiles",
    "percentiles",
    "show-histogram",
    "hist-bins",
    "preview-rows",
    "preview-cols",
    "preview-mode",
}


class Debug(Transformer):
    """Print selected frame diagnostics while passing packets through unchanged."""

    type_name = "debug"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in")},
            output_ports={"out": PortContract("out")},
            parameters={
                "enabled": ParameterContract(
                    "enabled",
                    "bool",
                    default=True,
                    description="Disable printing while still passing frames through.",
                ),
                "every-seconds": ParameterContract(
                    "every-seconds",
                    "float",
                    default="<print once>",
                    description="Print when frame pts advances by this many seconds.",
                ),
                "every-frames": ParameterContract(
                    "every-frames",
                    "int",
                    default="<print once>",
                    description="Print every N input frames.",
                ),
                "label": ParameterContract(
                    "label",
                    "str",
                    default="<element id>",
                    description="Label included in each debug record.",
                ),
                "stream": ParameterContract(
                    "stream",
                    "str",
                    default="stdout",
                    choices=tuple(sorted(_STREAMS)),
                    description="Output stream for debug records.",
                ),
                "path": ParameterContract(
                    "path",
                    "path",
                    default=None,
                    description="Append target when stream=file.",
                ),
                "show-shape": ParameterContract(
                    "show-shape", "bool", default=True, description="Print frame shape."
                ),
                "show-dtype": ParameterContract(
                    "show-dtype", "bool", default=True, description="Print frame dtype."
                ),
                "show-min": ParameterContract(
                    "show-min", "bool", default=True, description="Print frame minimum."
                ),
                "show-max": ParameterContract(
                    "show-max", "bool", default=True, description="Print frame maximum."
                ),
                "show-mean": ParameterContract(
                    "show-mean", "bool", default=True, description="Print frame mean."
                ),
                "show-std": ParameterContract(
                    "show-std",
                    "bool",
                    default=True,
                    description="Print frame standard deviation.",
                ),
                "show-median": ParameterContract(
                    "show-median",
                    "bool",
                    default=True,
                    description="Print frame median.",
                ),
                "show-preview": ParameterContract(
                    "show-preview",
                    "bool",
                    default=False,
                    description="Print bounded frame sample slices.",
                ),
                "show-percentiles": ParameterContract(
                    "show-percentiles",
                    "bool",
                    default=False,
                    description="Print configured frame percentiles.",
                ),
                "percentiles": ParameterContract(
                    "percentiles",
                    "list",
                    default="0.001,0.01,0.5,0.99,0.999",
                    description="Comma-separated percentile fractions.",
                ),
                "show-histogram": ParameterContract(
                    "show-histogram",
                    "bool",
                    default=False,
                    description="Print compact histogram counts.",
                ),
                "hist-bins": ParameterContract(
                    "hist-bins",
                    "int",
                    default=64,
                    description="Histogram bin count.",
                ),
                "preview-rows": ParameterContract(
                    "preview-rows",
                    "int",
                    default=4,
                    description="Maximum preview rows.",
                ),
                "preview-cols": ParameterContract(
                    "preview-cols",
                    "int",
                    default=8,
                    description="Maximum preview columns.",
                ),
                "preview-mode": ParameterContract(
                    "preview-mode",
                    "str",
                    default="both",
                    choices=tuple(sorted(_PREVIEW_MODES)),
                    description="Preview region selection.",
                ),
            },
            description="Print frame diagnostics and pass frames through unchanged.",
            subcategory="Debug",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized_params = _normalize_aliases(params)

        self.enabled = _bool(normalized_params.get("enabled", True))
        self.every_seconds = _optional_float(normalized_params, "every-seconds")
        self.every_frames = _optional_int(normalized_params, "every-frames")
        if self.every_seconds is not None and self.every_seconds <= 0:
            raise ValueError("debug every-seconds must be positive")
        if self.every_frames is not None and self.every_frames <= 0:
            raise ValueError("debug every-frames must be positive")
        if self.every_seconds is not None and self.every_frames is not None:
            raise ValueError("debug cannot combine every-seconds and every-frames")

        self.label = str(normalized_params.get("label", self.instance_id))
        self.stream = str(normalized_params.get("stream", "stdout"))
        if self.stream not in _STREAMS:
            raise ValueError("debug stream must be stdout, stderr, or file")
        path = normalized_params.get("path")
        self.path = str(path) if path is not None else None
        if self.stream == "file" and not self.path:
            raise ValueError("debug path is required when stream=file")

        self.show_shape = _bool(normalized_params.get("show-shape", True))
        self.show_dtype = _bool(normalized_params.get("show-dtype", True))
        self.show_min = _bool(normalized_params.get("show-min", True))
        self.show_max = _bool(normalized_params.get("show-max", True))
        self.show_mean = _bool(normalized_params.get("show-mean", True))
        self.show_std = _bool(normalized_params.get("show-std", True))
        self.show_median = _bool(normalized_params.get("show-median", True))
        self.show_preview = _bool(normalized_params.get("show-preview", False))
        self.show_percentiles = _bool(
            normalized_params.get("show-percentiles", False)
        )
        self.percentiles = _parse_float_list(
            normalized_params.get("percentiles", "0.001,0.01,0.5,0.99,0.999")
        )
        if any(percentile < 0.0 or percentile > 1.0 for percentile in self.percentiles):
            raise ValueError("debug percentiles must be in the range 0.0..1.0")
        self.show_histogram = _bool(normalized_params.get("show-histogram", False))
        self.hist_bins = int(normalized_params.get("hist-bins", 64))
        if self.hist_bins <= 0:
            raise ValueError("debug hist-bins must be positive")

        self.preview_rows = int(normalized_params.get("preview-rows", 4))
        self.preview_cols = int(normalized_params.get("preview-cols", 8))
        if self.preview_rows <= 0:
            raise ValueError("debug preview-rows must be positive")
        if self.preview_cols <= 0:
            raise ValueError("debug preview-cols must be positive")
        self.preview_mode = str(normalized_params.get("preview-mode", "both"))
        if self.preview_mode not in _PREVIEW_MODES:
            raise ValueError("debug preview-mode must be top-left, center, or both")

        self._frames_seen = 0
        self._printed_once = False
        self._last_print_pts: float | None = None
        self._file: TextIO | None = None

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        if self.enabled and self._should_print(packet):
            self._emit(self._format_packet(packet))
            self._printed_once = True
            self._last_print_pts = packet.metadata.pts
        self._frames_seen += 1
        return {"out": [packet]}

    def stop(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _should_print(self, packet: FramePacket) -> bool:
        if self.every_frames is not None:
            return self._frames_seen % self.every_frames == 0
        if self.every_seconds is not None:
            if self._last_print_pts is None:
                return True
            return packet.metadata.pts - self._last_print_pts >= self.every_seconds
        return not self._printed_once

    def _format_packet(self, packet: FramePacket) -> str:
        frame = packet.data
        lines = [
            (
                f"[debug {self.label}] index={packet.metadata.index} "
                f"pts={packet.metadata.pts:.6f}"
            )
        ]
        fields: list[str] = []
        if self.show_shape:
            fields.append(f"shape={frame.shape}")
        if self.show_dtype:
            fields.append(f"dtype={frame.dtype}")
        if fields:
            lines.append(" ".join(fields))

        stats: list[str] = []
        if frame.size == 0:
            if self.show_min:
                stats.append("min=n/a")
            if self.show_max:
                stats.append("max=n/a")
            if self.show_mean:
                stats.append("mean=n/a")
            if self.show_std:
                stats.append("std=n/a")
            if self.show_median:
                stats.append("median=n/a")
        else:
            if self.show_min:
                stats.append(f"min={np.min(frame)}")
            if self.show_max:
                stats.append(f"max={np.max(frame)}")
            if self.show_mean:
                stats.append(f"mean={float(np.mean(frame)):.6g}")
            if self.show_std:
                stats.append(f"std={float(np.std(frame)):.6g}")
            if self.show_median:
                stats.append(f"median={float(np.median(frame)):.6g}")
        if stats:
            lines.append(" ".join(stats))

        if frame.size and self.show_percentiles:
            values = [
                f"p{percentile:g}={float(np.quantile(frame, percentile)):.6g}"
                for percentile in self.percentiles
            ]
            lines.append("percentiles " + " ".join(values))
        if frame.size and self.show_histogram:
            hist, _ = np.histogram(frame, bins=self.hist_bins)
            lines.append(f"histogram bins={self.hist_bins}: {hist.tolist()}")
        if self.show_preview:
            lines.extend(self._format_previews(frame))
        return "\n".join(lines)

    def _format_previews(self, frame: np.ndarray) -> list[str]:
        previews: list[str] = []
        if self.preview_mode in {"top-left", "both"}:
            top_left = _slice_top_left(frame, self.preview_rows, self.preview_cols)
            previews.extend(
                [
                    f"preview top-left rows={self.preview_rows} cols={self.preview_cols}:",
                    np.array2string(top_left),
                ]
            )
        if self.preview_mode in {"center", "both"}:
            center = _slice_center(frame, self.preview_rows, self.preview_cols)
            previews.extend(
                [
                    f"preview center rows={self.preview_rows} cols={self.preview_cols}:",
                    np.array2string(center),
                ]
            )
        return previews

    def _emit(self, text: str) -> None:
        stream = self._output_stream()
        print(text, file=stream)
        stream.flush()

    def _output_stream(self) -> TextIO:
        if self.stream == "stdout":
            return sys.stdout
        if self.stream == "stderr":
            return sys.stderr
        if self._file is None:
            self._file = open(self.path or "", "a", encoding="utf-8")
        return self._file


def _normalize_aliases(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    for key in list(params):
        canonical = key.replace("_", "-")
        if canonical == key or canonical not in _PARAMETER_NAMES:
            continue
        if canonical in normalized:
            raise ValueError(f"debug cannot receive both {key!r} and {canonical!r}")
        normalized[canonical] = normalized.pop(key)
    return normalized


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return bool(value)


def _optional_float(params: dict[str, Any], name: str) -> float | None:
    value = params.get(name)
    if value is None:
        return None
    return float(value)


def _optional_int(params: dict[str, Any], name: str) -> int | None:
    value = params.get(name)
    if value is None:
        return None
    return int(value)


def _parse_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    return [float(value)]


def _slice_top_left(frame: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if frame.ndim == 0:
        return frame
    if frame.ndim == 1:
        return frame[:cols]
    return frame[:rows, :cols, ...]


def _slice_center(frame: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if frame.ndim == 0:
        return frame
    if frame.ndim == 1:
        start = max(0, (frame.shape[0] - cols) // 2)
        return frame[start : start + cols]
    row_start = max(0, (frame.shape[0] - rows) // 2)
    col_start = max(0, (frame.shape[1] - cols) // 2)
    return frame[row_start : row_start + rows, col_start : col_start + cols, ...]
