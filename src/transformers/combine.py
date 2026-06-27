"""Multi-input stream combiner."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FrameMetadata, FramePacket, infer_frame_shape, new_packet_id


_MODES = {"horizontal", "vertical", "grid"}


class Combine(Transformer):
    """Combine synchronized streams horizontally, vertically, or in a grid."""

    type_name = "combine"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            dynamic_input_ports={"in": PortContract("in")},
            output_ports={"out": PortContract("out")},
            parameters={
                "mode": ParameterContract(
                    "mode",
                    "str",
                    default="horizontal",
                    choices=tuple(sorted(_MODES)),
                    description="How to lay out connected input frames.",
                ),
                "rows": ParameterContract(
                    "rows",
                    "int",
                    default=None,
                    description="Grid row count; required when mode=grid.",
                ),
                "cols": ParameterContract(
                    "cols",
                    "int",
                    default=None,
                    description="Grid column count; required when mode=grid.",
                ),
                "stream_id": ParameterContract(
                    "stream_id",
                    "str",
                    default="<joined input stream ids>",
                    description="Output stream id override.",
                ),
            },
            description="Combine streams horizontally, vertically, or in a grid.",
            require_same_format=True,
            require_same_depth=True,
            require_same_index=True,
            synchronized_inputs=True,
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.mode = str(params.get("mode", "horizontal"))
        if self.mode not in _MODES:
            raise ValueError("combine mode must be horizontal, vertical, or grid")
        self.rows = int(params["rows"]) if params.get("rows") is not None else None
        self.cols = int(params["cols"]) if params.get("cols") is not None else None
        if self.rows is not None and self.rows <= 0:
            raise ValueError("combine rows must be a positive integer")
        if self.cols is not None and self.cols <= 0:
            raise ValueError("combine cols must be a positive integer")
        if self.mode == "grid" and (self.rows is None or self.cols is None):
            raise ValueError("combine grid mode requires rows and cols")
        self.stream_id = params.get("stream_id")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        indexed_packets = _indexed_packets(inputs)
        if not indexed_packets:
            raise ValueError("combine requires at least one connected input")
        packets = [packet for _, packet in indexed_packets]
        self._validate_packets(indexed_packets)

        if self.mode == "horizontal":
            combined = np.hstack([packet.data for packet in packets])
            missing_ports: list[str] = []
        elif self.mode == "vertical":
            combined = np.vstack([packet.data for packet in packets])
            missing_ports = []
        else:
            combined, missing_ports = self._grid(indexed_packets)

        metadata = self._metadata(
            packets,
            combined,
            [idx for idx, _ in indexed_packets],
            missing_ports,
        )
        return {"out": [FramePacket(data=combined, metadata=metadata)]}

    def _validate_packets(
        self, indexed_packets: list[tuple[int, FramePacket]]
    ) -> None:
        if self.mode == "grid":
            assert self.rows is not None
            assert self.cols is not None
            slot_count = self.rows * self.cols
            for index, _ in indexed_packets:
                if index >= slot_count:
                    raise ValueError(
                        f"combine grid input in{index} exceeds rows*cols={slot_count}"
                    )

        first = indexed_packets[0][1].metadata
        for _, packet in indexed_packets[1:]:
            metadata = packet.metadata
            if metadata.format != first.format:
                raise ValueError("combine requires matching frame formats")
            if metadata.depth != first.depth:
                raise ValueError("combine requires matching frame depths")
            if metadata.channels != first.channels:
                raise ValueError("combine requires matching channel counts")
            if metadata.index != first.index:
                raise ValueError("combine requires matching frame indexes")

            if self.mode == "horizontal" and metadata.height != first.height:
                raise ValueError("horizontal combine requires equal heights")
            if self.mode == "vertical" and metadata.width != first.width:
                raise ValueError("vertical combine requires equal widths")
            if self.mode == "grid" and (
                metadata.width != first.width or metadata.height != first.height
            ):
                raise ValueError("grid combine requires equal dimensions")

    def _grid(
        self, indexed_packets: list[tuple[int, FramePacket]]
    ) -> tuple[np.ndarray, list[str]]:
        assert self.rows is not None
        assert self.cols is not None
        packets_by_index = dict(indexed_packets)
        template = indexed_packets[0][1].data
        missing_ports: list[str] = []
        rows: list[np.ndarray] = []
        for row in range(self.rows):
            cells: list[np.ndarray] = []
            for col in range(self.cols):
                index = row * self.cols + col
                packet = packets_by_index.get(index)
                if packet is None:
                    missing_ports.append(f"in{index}")
                    cells.append(np.zeros_like(template))
                else:
                    cells.append(packet.data)
            rows.append(np.hstack(cells))
        return np.vstack(rows), missing_ports

    def _metadata(
        self,
        packets: list[FramePacket],
        combined: np.ndarray,
        indexes: list[int],
        missing_ports: list[str],
    ) -> FrameMetadata:
        width, height, channels, depth = infer_frame_shape(combined)
        parents = tuple(
            dict.fromkeys(
                parent
                for packet in packets
                for parent in (*packet.metadata.parents, packet.metadata.packet_id)
            )
        )
        first = packets[0].metadata
        stream_ids = [packet.metadata.stream_id for packet in packets]
        source_ids = [packet.metadata.source_id for packet in packets]
        return FrameMetadata(
            packet_id=new_packet_id(),
            stream_id=str(self.stream_id or "+".join(stream_ids)),
            source_id="+".join(source_ids),
            pts=max(packet.metadata.pts for packet in packets),
            index=first.index,
            format=first.format,
            width=width,
            height=height,
            fps=min(packet.metadata.fps for packet in packets),
            depth=depth,
            channels=channels,
            parents=parents,
            extra={
                "combined_by": self.instance_id,
                "combine_mode": self.mode,
                "input_ports": [f"in{index}" for index in indexes],
                "input_stream_ids": stream_ids,
                "grid_rows": self.rows if self.mode == "grid" else None,
                "grid_cols": self.cols if self.mode == "grid" else None,
                "missing_input_ports": missing_ports,
            },
        )


def _indexed_packets(inputs: PacketInputs) -> list[tuple[int, FramePacket]]:
    indexed_packets: list[tuple[int, FramePacket]] = []
    for port_name, packet_or_list in inputs.items():
        index = _port_index(port_name)
        if isinstance(packet_or_list, list):
            if len(packet_or_list) != 1:
                raise ValueError(f"Port {port_name!r} expected one packet")
            packet = packet_or_list[0]
        else:
            packet = packet_or_list
        if not isinstance(packet, FramePacket):
            raise TypeError(f"Port {port_name!r} must receive a FramePacket")
        indexed_packets.append((index, packet))
    return sorted(indexed_packets, key=lambda item: item[0])


def _port_index(port_name: str) -> int:
    if not port_name.startswith("in") or not port_name[2:].isdigit():
        raise ValueError(f"combine input port must be named inN, got {port_name!r}")
    return int(port_name[2:])
