"""Bypass transformer."""

from __future__ import annotations

from src.lib.contracts import ElementContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer


class Bypass(Transformer):
    """Pass frames through unchanged."""

    type_name = "bypass"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={"in": PortContract("in")},
            output_ports={"out": PortContract("out")},
            description="Pass frames through unchanged.",
            subcategory="Control",
        )

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        return {"out": [self._single_input(inputs)]}
