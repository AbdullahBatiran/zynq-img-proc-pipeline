"""Base classes for graph elements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from .contracts import ElementContract
from .packets import FramePacket


PacketInputs = dict[str, FramePacket | list[FramePacket]]
PacketOutputs = dict[str, list[FramePacket]]


@dataclass
class PipelineContext:
    run_name: str = "default"
    extra: dict[str, Any] = field(default_factory=dict)


class Element:
    type_name: ClassVar[str] = "element"

    def __init__(self, instance_id: str, params: dict[str, Any] | None = None) -> None:
        self.instance_id = instance_id
        self.params: dict[str, Any] = {}
        self.configure(params or {})

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(description=cls.__doc__ or "")

    def configure(self, params: dict[str, Any]) -> None:
        self.params = dict(params)

    def start(self, context: PipelineContext) -> None:
        pass

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        raise NotImplementedError

    def stop(self) -> None:
        pass

    def _single_input(self, inputs: PacketInputs, port: str = "in") -> FramePacket:
        packet_or_list = inputs.get(port)
        if isinstance(packet_or_list, list):
            if len(packet_or_list) != 1:
                raise ValueError(f"Port {port!r} expected one packet")
            packet = packet_or_list[0]
        else:
            packet = packet_or_list
        if not isinstance(packet, FramePacket):
            raise TypeError(f"Port {port!r} must receive a FramePacket")
        return packet


class Source(Element):
    pass


class Transformer(Element):
    pass


class Sink(Element):
    pass
