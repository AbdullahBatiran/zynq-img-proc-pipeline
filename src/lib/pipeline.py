"""Graph pipeline specification, validation, and synchronous execution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .contracts import validate_packets_are_frame_packets
from .elements import Element, PipelineContext, Sink, Source
from .packets import FramePacket
from .registry import ElementRegistry, default_registry, register_builtin_elements


@dataclass(frozen=True)
class ElementSpec:
    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectionSpec:
    from_element: str
    from_port: str
    to_element: str
    to_port: str


@dataclass(frozen=True)
class PipelineSpec:
    elements: list[ElementSpec]
    connections: list[ConnectionSpec]


class Pipeline:
    def __init__(
        self,
        spec: PipelineSpec,
        registry: ElementRegistry | None = None,
        context: PipelineContext | None = None,
    ) -> None:
        self.spec = spec
        self.registry = registry or default_registry
        self.context = context or PipelineContext()
        self.elements: dict[str, Element] = {}
        self.adjacency: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        self.incoming_ports: dict[str, set[str]] = defaultdict(set)
        self.outgoing_ports: dict[str, set[str]] = defaultdict(set)

    @classmethod
    def from_spec(
        cls,
        spec: PipelineSpec,
        registry: ElementRegistry | None = None,
        context: PipelineContext | None = None,
    ) -> "Pipeline":
        register_builtin_elements(registry or default_registry)
        pipeline = cls(spec=spec, registry=registry, context=context)
        pipeline.build()
        pipeline.validate()
        return pipeline

    def build(self) -> None:
        ids: set[str] = set()
        for element_spec in self.spec.elements:
            if element_spec.id in ids:
                raise ValueError(f"Duplicate element id {element_spec.id!r}")
            ids.add(element_spec.id)
            self.elements[element_spec.id] = self.registry.create(
                element_spec.type, element_spec.id, element_spec.params
            )
        for connection in self.spec.connections:
            self.adjacency[(connection.from_element, connection.from_port)].append(
                (connection.to_element, connection.to_port)
            )
            self.incoming_ports[connection.to_element].add(connection.to_port)
            self.outgoing_ports[connection.from_element].add(connection.from_port)
        for element_id, element in self.elements.items():
            element.configure_connected_output_ports(
                set(self.outgoing_ports.get(element_id, set()))
            )

    def validate(self) -> None:
        if not self.elements:
            raise ValueError("Pipeline has no elements")

        for connection in self.spec.connections:
            if connection.from_element not in self.elements:
                raise ValueError(f"Unknown source element {connection.from_element!r}")
            if connection.to_element not in self.elements:
                raise ValueError(f"Unknown target element {connection.to_element!r}")

            source_contract = self.elements[connection.from_element].contract()
            target_contract = self.elements[connection.to_element].contract()
            source_port = source_contract.output_contract(connection.from_port)
            if source_port is None:
                raise ValueError(
                    f"Element {connection.from_element!r} has no output port "
                    f"{connection.from_port!r}"
                )
            target_port = target_contract.input_contract(connection.to_port)
            if target_port is None:
                raise ValueError(
                    f"Element {connection.to_element!r} has no input port "
                    f"{connection.to_port!r}"
                )

            out_port = source_port
            in_port = target_port
            if in_port.formats is not None and out_port.formats is not None:
                if not in_port.formats.intersection(out_port.formats):
                    raise ValueError(
                        f"Incompatible formats from {connection.from_element}."
                        f"{connection.from_port} to {connection.to_element}."
                        f"{connection.to_port}"
                    )
            if in_port.depths is not None and out_port.depths is not None:
                if not in_port.depths.intersection(out_port.depths):
                    raise ValueError(
                        f"Incompatible depths from {connection.from_element}."
                        f"{connection.from_port} to {connection.to_element}."
                        f"{connection.to_port}"
                    )

    def run(self, max_frames: int | None = None) -> None:
        self._start_all()
        buffers: dict[tuple[str, str], list[FramePacket]] = defaultdict(list)
        source_ids = [
            element_id
            for element_id, element in self.elements.items()
            if isinstance(element, Source)
        ]
        active_sources = set(source_ids)
        source_counts = {source_id: 0 for source_id in source_ids}

        try:
            while (
                not self.context.stop_requested
                and (active_sources or self._has_buffered_packets(buffers))
            ):
                for source_id in list(active_sources):
                    if self.context.stop_requested:
                        break
                    if max_frames is not None and source_counts[source_id] >= max_frames:
                        active_sources.remove(source_id)
                        continue

                    outputs = self.elements[source_id].process({})
                    if not outputs:
                        active_sources.remove(source_id)
                        continue

                    self._route_outputs(source_id, outputs, buffers)
                    source_counts[source_id] += 1

                progressed = True
                processed_any = False
                while progressed and not self.context.stop_requested:
                    progressed = self._process_ready_elements(buffers)
                    processed_any = processed_any or progressed
                if (
                    not active_sources
                    and self._has_buffered_packets(buffers)
                    and not processed_any
                ):
                    raise RuntimeError(
                        "Pipeline stopped with buffered packets that cannot be "
                        "processed; check stream lengths and joiner compatibility"
                    )
        finally:
            self._stop_all()

    def _start_all(self) -> None:
        for element in self.elements.values():
            element.start(self.context)

    def _stop_all(self) -> None:
        for element in reversed(list(self.elements.values())):
            element.stop()

    def _process_ready_elements(
        self, buffers: dict[tuple[str, str], list[FramePacket]]
    ) -> bool:
        progressed = False
        for element_id, element in self.elements.items():
            if isinstance(element, Source):
                continue
            input_ports = self._required_input_ports(element_id, element)
            if not input_ports:
                continue
            if not all(buffers[(element_id, port)] for port in input_ports):
                continue

            inputs: dict[str, FramePacket] = {
                port: buffers[(element_id, port)].pop(0) for port in input_ports
            }
            self._validate_inputs(element, inputs)
            outputs = element.process(inputs)
            self._route_outputs(element_id, outputs, buffers)
            progressed = True
        return progressed

    def _required_input_ports(self, element_id: str, element: Element) -> set[str]:
        contract = element.contract()
        contract_ports = contract.input_names()
        connected_ports = self.incoming_ports.get(element_id, set())
        dynamic_ports = contract.dynamic_input_names(connected_ports)
        required_static_ports = contract_ports.intersection(connected_ports)
        return required_static_ports | dynamic_ports or contract_ports

    def _validate_inputs(
        self, element: Element, inputs: dict[str, FramePacket]
    ) -> None:
        contract = element.contract()
        packets = list(inputs.values())
        validate_packets_are_frame_packets(packets)
        for port_name, packet in inputs.items():
            port_contract = contract.input_contract(port_name)
            if port_contract is None:
                raise ValueError(f"{element.instance_id} has no input port {port_name!r}")
            port_contract.validate_packet(packet)

        if len(packets) < 2:
            return
        first = packets[0].metadata
        for packet in packets[1:]:
            metadata = packet.metadata
            if contract.require_same_size and (
                metadata.width != first.width or metadata.height != first.height
            ):
                raise ValueError(f"{element.instance_id} requires equal input sizes")
            if contract.require_same_format and metadata.format != first.format:
                raise ValueError(f"{element.instance_id} requires equal formats")
            if contract.require_same_depth and metadata.depth != first.depth:
                raise ValueError(f"{element.instance_id} requires equal depths")
            if contract.require_same_index and metadata.index != first.index:
                raise ValueError(f"{element.instance_id} requires equal frame indexes")
            if contract.require_same_pts and metadata.pts != first.pts:
                raise ValueError(f"{element.instance_id} requires equal timestamps")

    def _route_outputs(
        self,
        element_id: str,
        outputs: dict[str, list[FramePacket]],
        buffers: dict[tuple[str, str], list[FramePacket]],
    ) -> None:
        for port_name, packets in outputs.items():
            validate_packets_are_frame_packets(packets)
            contract = self.elements[element_id].contract()
            output_port = contract.output_contract(port_name)
            if output_port is None:
                raise ValueError(f"{element_id!r} has no output port {port_name!r}")
            for packet in packets:
                output_port.validate_packet(packet)
            for target_element, target_port in self.adjacency.get(
                (element_id, port_name), []
            ):
                buffers[(target_element, target_port)].extend(packets)

    @staticmethod
    def _has_buffered_packets(
        buffers: dict[tuple[str, str], list[FramePacket]]
    ) -> bool:
        return any(bool(packets) for packets in buffers.values())
