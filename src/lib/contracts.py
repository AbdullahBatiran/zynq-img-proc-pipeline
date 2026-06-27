"""Element port contracts and compatibility validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .packets import FramePacket


REQUIRED_METADATA_FIELDS = (
    "packet_id",
    "stream_id",
    "source_id",
    "pts",
    "index",
    "format",
    "width",
    "height",
    "fps",
    "depth",
    "channels",
    "parents",
    "extra",
)


@dataclass(frozen=True)
class PortContract:
    name: str
    packet_type: type = FramePacket
    formats: set[str] | None = None
    depths: set[int] | None = None
    required_metadata: tuple[str, ...] = REQUIRED_METADATA_FIELDS

    def validate_packet(self, packet: FramePacket) -> None:
        if not isinstance(packet, self.packet_type):
            raise TypeError(f"Port {self.name!r} expected {self.packet_type.__name__}")
        metadata = packet.metadata
        for field_name in self.required_metadata:
            if not hasattr(metadata, field_name):
                raise ValueError(f"Packet metadata missing {field_name!r}")
        if self.formats is not None and metadata.format not in self.formats:
            raise ValueError(
                f"Port {self.name!r} does not accept format {metadata.format!r}"
            )
        if self.depths is not None and metadata.depth not in self.depths:
            raise ValueError(
                f"Port {self.name!r} does not accept depth {metadata.depth!r}"
            )


@dataclass(frozen=True)
class ElementContract:
    input_ports: dict[str, PortContract] = field(default_factory=dict)
    output_ports: dict[str, PortContract] = field(default_factory=dict)
    description: str = ""
    require_same_size: bool = False
    require_same_format: bool = False
    require_same_depth: bool = False
    require_same_index: bool = False
    require_same_pts: bool = False
    synchronized_inputs: bool = False

    def input_names(self) -> set[str]:
        return set(self.input_ports)

    def output_names(self) -> set[str]:
        return set(self.output_ports)


def validate_packets_are_frame_packets(packets: Iterable[FramePacket]) -> None:
    for packet in packets:
        if not isinstance(packet, FramePacket):
            raise TypeError("Pipeline elements must move FramePacket objects only")
