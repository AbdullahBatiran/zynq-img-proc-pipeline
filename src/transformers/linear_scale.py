"""Linear intensity scaling transformer."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lib.contracts import ElementContract, ParameterContract, PortContract
from src.lib.elements import PacketInputs, PacketOutputs, Transformer
from src.lib.packets import FramePacket, infer_frame_shape


_DTYPES: dict[str, type[np.unsignedinteger[Any]]] = {
    "uint8": np.uint8,
    "uint16": np.uint16,
}


class LinearScale(Transformer):
    """Linearly map an input intensity range to an output range."""

    type_name = "linear-scale"

    @classmethod
    def contract(cls) -> ElementContract:
        return ElementContract(
            input_ports={
                "in": PortContract(
                    "in", formats={"bgr", "rgb", "gray"}, depths={8, 16}
                )
            },
            output_ports={"out": PortContract("out", depths={8, 16})},
            parameters={
                "otype": ParameterContract(
                    "otype",
                    "str",
                    default="<same as input>",
                    choices=tuple(_DTYPES),
                    description="Output dtype.",
                ),
                "omin": ParameterContract(
                    "omin",
                    "number",
                    default=0,
                    description="Output intensity mapped from input minimum.",
                ),
                "omax": ParameterContract(
                    "omax",
                    "number",
                    default="<dtype max>",
                    description="Output intensity mapped from input maximum.",
                ),
                "min": ParameterContract(
                    "min",
                    "number",
                    default="<frame min>",
                    description="Input minimum override.",
                ),
                "max": ParameterContract(
                    "max",
                    "number",
                    default="<frame max>",
                    description="Input maximum override.",
                ),
                "perc": ParameterContract(
                    "perc",
                    "float",
                    default=None,
                    description="Symmetric lower and upper percentile clipping fraction.",
                ),
                "perc-down": ParameterContract(
                    "perc-down",
                    "float",
                    default=None,
                    description="Lower percentile clipping fraction.",
                ),
                "perc-up": ParameterContract(
                    "perc-up",
                    "float",
                    default=None,
                    description="Upper percentile clipping fraction.",
                ),
            },
            description="Linearly scale frame intensities.",
        )

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        normalized_params = _normalize_aliases(params)

        self.otype = normalized_params.get("otype")
        if self.otype is not None:
            self.otype = str(self.otype)
            if self.otype not in _DTYPES:
                raise ValueError("linear-scale otype must be 'uint8' or 'uint16'")

        self.omin = _optional_float(normalized_params, "omin")
        self.omax = _optional_float(normalized_params, "omax")
        self.input_min = _optional_float(normalized_params, "min")
        self.input_max = _optional_float(normalized_params, "max")
        self.perc = _optional_float(normalized_params, "perc")
        self.perc_down = _optional_float(normalized_params, "perc-down")
        self.perc_up = _optional_float(normalized_params, "perc-up")

        self._validate_percentile("perc", self.perc)
        self._validate_percentile("perc-down", self.perc_down)
        self._validate_percentile("perc-up", self.perc_up)

        if self.input_min is not None and (
            self.perc is not None or self.perc_down is not None
        ):
            raise ValueError("linear-scale cannot combine min with perc or perc-down")
        if self.input_max is not None and (
            self.perc is not None or self.perc_up is not None
        ):
            raise ValueError("linear-scale cannot combine max with perc or perc-up")
        if self.perc is not None and (
            self.perc_down is not None or self.perc_up is not None
        ):
            raise ValueError("linear-scale cannot combine perc with perc-down or perc-up")

    def process(self, inputs: PacketInputs) -> PacketOutputs:
        packet = self._single_input(inputs)
        self._validate_packet(packet)

        output_dtype = (
            _DTYPES[self.otype] if self.otype is not None else packet.data.dtype
        )
        output_max = _max_value_for_dtype(np.dtype(output_dtype))
        output_min_bound = self.omin if self.omin is not None else 0.0
        output_max_bound = self.omax if self.omax is not None else float(output_max)
        if output_min_bound >= output_max_bound:
            raise ValueError("linear-scale omin must be less than omax")

        input_min, input_max = self._input_range(packet.data)
        if input_min >= input_max:
            raise ValueError("linear-scale input min must be less than input max")

        scaled = (
            (packet.data.astype(np.float64) - input_min)
            * (output_max_bound - output_min_bound)
            / (input_max - input_min)
            + output_min_bound
        )
        scaled = np.clip(scaled, output_min_bound, output_max_bound)
        scaled = np.rint(scaled).astype(output_dtype)

        width, height, channels, depth = infer_frame_shape(scaled)
        metadata = packet.metadata.derive(
            width=width,
            height=height,
            channels=channels,
            depth=depth,
            extra={
                **packet.metadata.extra,
                "linear_scaled_by": self.instance_id,
                "linear_scale_input_min": input_min,
                "linear_scale_input_max": input_max,
                "linear_scale_output_min": output_min_bound,
                "linear_scale_output_max": output_max_bound,
                "linear_scale_output_type": np.dtype(output_dtype).name,
            },
        )
        return {"out": [FramePacket(data=scaled, metadata=metadata)]}

    def _validate_packet(self, packet: FramePacket) -> None:
        metadata = packet.metadata
        if metadata.format not in {"bgr", "rgb", "gray"}:
            raise ValueError(f"linear-scale does not support format {metadata.format!r}")
        if metadata.depth not in {8, 16}:
            raise ValueError("linear-scale supports only 8-bit and 16-bit frames")
        if metadata.channels not in {1, 3}:
            raise ValueError("linear-scale supports only 1-channel or 3-channel frames")
        if packet.data.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
            raise ValueError("linear-scale supports only uint8 and uint16 frames")
        expected_dtype = np.uint8 if metadata.depth == 8 else np.uint16
        if packet.data.dtype != expected_dtype:
            raise ValueError(
                f"linear-scale expected dtype {expected_dtype} for "
                f"{metadata.depth}-bit metadata"
            )
        if metadata.channels == 1 and packet.data.ndim not in {2, 3}:
            raise ValueError("1-channel frames must be 2D or HxWx1 arrays")
        if metadata.channels == 3 and (
            packet.data.ndim != 3 or packet.data.shape[2] != 3
        ):
            raise ValueError("3-channel frames must be HxWx3 arrays")

    def _input_range(self, frame: np.ndarray) -> tuple[float, float]:
        if self.input_min is not None:
            input_min = self.input_min
        elif self.perc is not None:
            input_min = _percentile(frame, self.perc)
        elif self.perc_down is not None:
            input_min = _percentile(frame, self.perc_down)
        else:
            input_min = float(np.min(frame))

        if self.input_max is not None:
            input_max = self.input_max
        elif self.perc is not None:
            input_max = _percentile(frame, 1.0 - self.perc)
        elif self.perc_up is not None:
            input_max = _percentile(frame, 1.0 - self.perc_up)
        else:
            input_max = float(np.max(frame))

        return input_min, input_max

    @staticmethod
    def _validate_percentile(name: str, value: float | None) -> None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"linear-scale {name} must be in the range 0.0..1.0")


def _normalize_aliases(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    for alias, canonical in (("perc_down", "perc-down"), ("perc_up", "perc-up")):
        if alias in normalized:
            if canonical in normalized:
                raise ValueError(
                    f"linear-scale cannot receive both {alias!r} and {canonical!r}"
                )
            normalized[canonical] = normalized.pop(alias)
    return normalized


def _optional_float(params: dict[str, Any], name: str) -> float | None:
    value = params.get(name)
    if value is None:
        return None
    return float(value)


def _percentile(frame: np.ndarray, fraction: float) -> float:
    return float(np.quantile(frame, fraction))


def _max_value_for_dtype(dtype: np.dtype[Any]) -> int:
    if dtype == np.dtype(np.uint8):
        return 255
    if dtype == np.dtype(np.uint16):
        return 65535
    raise ValueError("linear-scale supports only uint8 and uint16 frames")
