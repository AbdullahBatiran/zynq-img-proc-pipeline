"""Interlace mimic test transformer."""

from __future__ import annotations

from typing import Any

from src.lib.contracts import ElementContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, PipelineContext, Transformer
from src.lib.packets import FramePacket


_LINE_ORDER = "first_rows_0_even_second_rows_1_odd"


class InterlaceMimicTest(Transformer):
    """Combine pairs of frames by alternating rows from first and second inputs."""

    type_name = "interlace-mimic-test"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in")},
            output_ports={"out": PortContract("out")},
            description="Combine frame pairs by alternating rows.",
            subcategory="Test",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self._pending: FramePacket | None = None
        self._pair_index = 0

    def start(self, context: PipelineContext) -> None:
        self._pending = None
        self._pair_index = 0

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        if self._pending is None:
            self._pending = packet
            return {}

        first = self._pending
        second = packet
        self._pending = None
        self._validate_pair(first, second)

        interlaced = first.data.copy()
        interlaced[1::2] = second.data[1::2]

        pair_index = self._pair_index
        self._pair_index += 1
        first_metadata = first.metadata
        second_metadata = second.metadata
        parents = tuple(
            dict.fromkeys(
                (
                    *first_metadata.parents,
                    first_metadata.packet_id,
                    *second_metadata.parents,
                    second_metadata.packet_id,
                )
            )
        )
        fps = first_metadata.fps / 2.0 if first_metadata.fps > 0 else first_metadata.fps
        metadata = first_metadata.derive(
            index=pair_index,
            pts=first_metadata.pts,
            fps=fps,
            parents=parents,
            extra={
                **first_metadata.extra,
                "interlace_mimic_test_by": self.instance_id,
                "interlace_mimic_pair_index": pair_index,
                "interlace_mimic_first_index": first_metadata.index,
                "interlace_mimic_second_index": second_metadata.index,
                "interlace_mimic_line_order": _LINE_ORDER,
            },
        )
        return {"out": [FramePacket(data=interlaced, metadata=metadata)]}

    def stop(self) -> None:
        self._pending = None

    def _validate_pair(self, first: FramePacket, second: FramePacket) -> None:
        if first.data.shape != second.data.shape:
            raise ValueError("interlace-mimic-test requires matching frame shapes")
        if first.data.dtype != second.data.dtype:
            raise ValueError("interlace-mimic-test requires matching frame dtypes")

        first_metadata = first.metadata
        second_metadata = second.metadata
        if first_metadata.width != second_metadata.width:
            raise ValueError("interlace-mimic-test requires matching widths")
        if first_metadata.height != second_metadata.height:
            raise ValueError("interlace-mimic-test requires matching heights")
        if first_metadata.channels != second_metadata.channels:
            raise ValueError("interlace-mimic-test requires matching channel counts")
        if first_metadata.format != second_metadata.format:
            raise ValueError("interlace-mimic-test requires matching formats")
        if first_metadata.depth != second_metadata.depth:
            raise ValueError("interlace-mimic-test requires matching depths")
