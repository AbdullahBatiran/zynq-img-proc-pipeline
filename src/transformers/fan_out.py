"""Fan out one input stream to multiple output ports."""

from __future__ import annotations

from typing import Any

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket


class FanOut(Transformer):
    """Replicate one input packet to connected output ports."""

    type_name = "fan-out"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in")},
            dynamic_output_ports={"out": PortContract("out")},
            parameters={
                "outputs": ParameterContract(
                    "outputs",
                    "int",
                    default="<connected outputs>",
                    description="Optional number of outN ports to emit.",
                ),
            },
            description="Replicate one input stream to multiple output ports.",
            subcategory="Control",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self.output_count = (
            int(params["outputs"]) if params.get("outputs") is not None else None
        )
        if self.output_count is not None and self.output_count <= 0:
            raise ValueError("fan-out outputs must be a positive integer")
        self.connected_output_ports: set[str] = set()

    def configure_connected_output_ports(self, ports: set[str]) -> None:
        for port_name in ports:
            index = _output_index(port_name)
            if self.output_count is not None and index >= self.output_count:
                raise ValueError(
                    f"fan-out output port {port_name!r} exceeds outputs="
                    f"{self.output_count}"
                )
        self.connected_output_ports = set(ports)

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        return {port_name: [packet] for port_name in self._output_ports()}

    def _output_ports(self) -> list[str]:
        if self.output_count is not None:
            return [f"out{index}" for index in range(self.output_count)]
        if self.connected_output_ports:
            return sorted(self.connected_output_ports, key=_output_index)
        return ["out0"]


def _output_index(port_name: str) -> int:
    if not port_name.startswith("out") or not port_name[3:].isdigit():
        raise ValueError(f"fan-out output port must be named outN, got {port_name!r}")
    return int(port_name[3:])
