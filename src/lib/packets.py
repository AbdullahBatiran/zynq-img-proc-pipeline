"""Frame packet and metadata types.

Every runtime payload in the pipeline must move as a FramePacket. This keeps
frame data attached to compatibility metadata and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

import numpy as np


def new_packet_id() -> str:
    return uuid4().hex


@dataclass(frozen=True)
class FrameMetadata:
    packet_id: str
    stream_id: str
    source_id: str
    pts: float
    index: int
    format: str
    width: int
    height: int
    fps: float
    depth: int
    channels: int
    parents: tuple[str, ...] = field(default_factory=tuple)
    extra: dict[str, Any] = field(default_factory=dict)

    def derive(self, **changes: Any) -> "FrameMetadata":
        """Create metadata for a derived frame and preserve provenance."""
        parents = changes.pop("parents", None)
        if parents is None:
            parents = (*self.parents, self.packet_id)
        extra = changes.pop("extra", None)
        if extra is None:
            extra = dict(self.extra)
        return replace(
            self,
            packet_id=new_packet_id(),
            parents=tuple(dict.fromkeys(parents)),
            extra=extra,
            **changes,
        )


@dataclass(frozen=True)
class FramePacket:
    data: np.ndarray
    metadata: FrameMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FrameMetadata):
            raise TypeError("FramePacket metadata must be a FrameMetadata instance")
        if not isinstance(self.data, np.ndarray):
            raise TypeError("FramePacket data must be a numpy.ndarray")


def infer_frame_shape(data: np.ndarray) -> tuple[int, int, int, int]:
    """Return width, height, channels, depth from a NumPy frame."""
    if data.ndim == 2:
        height, width = data.shape
        channels = 1
    elif data.ndim == 3:
        height, width, channels = data.shape
    else:
        raise ValueError(f"Unsupported frame shape: {data.shape}")
    depth = data.dtype.itemsize * 8
    return width, height, channels, depth
