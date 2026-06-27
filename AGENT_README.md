# Zynq Image Processing Pipeline Project Guide

This project is a Python video-stream processing framework built around small
graph elements connected by named ports. It is inspired by GStreamer-style
pipelines, but the implementation is intentionally compact and synchronous so
new image-processing behavior can be added and tested quickly.

This guide is for agents and contributors who need to understand the internal
pipeline model. It does not describe individual built-in elements; use the CLI
introspection commands for that.

## Repository Layout

```text
src/
  cli.py                    CLI entrypoint and user-facing formatting.
  lib/
    packets.py              FramePacket and FrameMetadata definitions.
    contracts.py            Port, parameter, and element contract metadata.
    elements.py             Element base classes and lifecycle hooks.
    registry.py             Built-in element registration and lookup.
    pipeline.py             PipelineSpec, validation, scheduler, routing.
    cli_parse.py            GStreamer-like pipeline expression parser.
    opencv_qt.py            OpenCV/Qt environment setup helper.
  sources/                  Source element implementations.
  transformers/             Transform and control element implementations.
    control/                Placeholder package for control-oriented modules.
    gui/                    Placeholder package for GUI-oriented modules.
  sinks/                    Sink element implementations.
tests/
  test_pipeline.py          Unit and CLI smoke coverage.
examples/                   Small usage examples.
saved_commands/             Local command scripts; not required by the library.
data/                       Local sample data; not required by the library.
```

## Runtime Data Model

All runtime frame payloads move through the graph as `FramePacket` objects.
Elements should not pass raw NumPy arrays between ports.

`FramePacket` contains:

- `data`: the NumPy frame array.
- `metadata`: a `FrameMetadata` object describing the frame and provenance.

`FrameMetadata` contains stream identity, timing, frame dimensions, pixel
format/depth/channel information, parent packet IDs, and an extensible `extra`
dictionary. The `derive()` helper creates a new metadata object while preserving
provenance by adding the current packet ID to `parents`.

Use `metadata.derive(...)` when an element creates a new derived frame. If an
element passes the exact input packet through unchanged, it should return that
same `FramePacket` object rather than cloning metadata.

## Core Library Concepts

### Elements

Every runnable node is an `Element`. The three concrete roles are:

- `Source`: produces packets and has no required input stream.
- `Transformer`: consumes packets and emits packets.
- `Sink`: consumes packets and normally emits no downstream packets.

Elements have these lifecycle hooks:

```python
def configure(self, params: dict[str, Any]) -> None:
    ...

def configure_connected_output_ports(self, ports: set[str]) -> None:
    ...

def start(self, context: PipelineContext) -> None:
    ...

def process(self, inputs: PacketInputs) -> PacketOutputs:
    ...

def stop(self) -> None:
    ...
```

`configure()` runs during element construction. `start()` runs once before the
pipeline loop. `process()` runs whenever the scheduler has the required inputs
for the element. `stop()` runs in reverse element order when the pipeline ends.

`configure_connected_output_ports()` runs after graph connections are built and
before validation. Elements with dynamic output ports can use this hook to learn
which output ports are actually connected.

### Context

`PipelineContext` carries shared run state:

- `run_name`
- `extra`
- `stop_requested`

Sinks or control elements can call `context.request_stop()` to ask the scheduler
to end the run.

### Contracts

An element exposes its interface through `ElementContract`.

The contract defines:

- Static input ports.
- Dynamic input port prefixes.
- Static output ports.
- Dynamic output port prefixes.
- Parameter metadata.
- Human-readable description and optional subcategory.
- Compatibility rules such as matching format, depth, index, timestamp, or size.
- Whether multiple inputs are synchronized.

`PortContract` validates packet type and optional metadata constraints such as
accepted formats or depths. `ParameterContract` describes user-facing parameters
for CLI introspection and documentation.

Dynamic ports use a prefix plus a numeric suffix. For example, a dynamic input
prefix of `in` accepts `in0`, `in1`, `in2`, and so on. A dynamic output prefix
of `out` accepts `out0`, `out1`, `out2`, and so on.

### Registry

`ElementRegistry` maps element type names to Python classes. The default
registry is populated by `register_builtin_elements()`.

When adding a built-in element, import the class in `src/lib/registry.py` and
include it in the registration tuple. Registered elements become available to:

- Python `PipelineSpec` construction.
- CLI pipeline expressions.
- `list-elements`.
- `describe`.

## Pipeline Specification

The graph is represented by three dataclasses in `src/lib/pipeline.py`:

```python
ElementSpec(id: str, type: str, params: dict[str, Any])
ConnectionSpec(from_element: str, from_port: str, to_element: str, to_port: str)
PipelineSpec(elements: list[ElementSpec], connections: list[ConnectionSpec])
```

`ElementSpec.id` is the instance name inside one graph. `ElementSpec.type` is the
registered element type. Connections always refer to instance IDs and named
ports.

Python code can construct a graph directly:

```python
from src.lib.pipeline import ConnectionSpec, ElementSpec, Pipeline, PipelineSpec

spec = PipelineSpec(
    elements=[
        ElementSpec(id="src", type="my-source", params={"path": "input.dat"}),
        ElementSpec(id="step", type="my-transform", params={}),
        ElementSpec(id="sink", type="my-sink", params={"path": "output.dat"}),
    ],
    connections=[
        ConnectionSpec("src", "out", "step", "in"),
        ConnectionSpec("step", "out", "sink", "in"),
    ],
)

Pipeline.from_spec(spec).run()
```

The names above are illustrative. Use `zpipe list-elements` to inspect the
actual registered element types in the current checkout.

## Pipeline Build And Validation

`Pipeline.from_spec(...)` does three things:

1. Ensures built-ins are registered.
2. Builds element instances and adjacency maps.
3. Validates the graph.

During `build()`:

- Duplicate element IDs are rejected.
- Each `ElementSpec` becomes an instantiated element.
- `adjacency[(from_element, from_port)]` records downstream targets.
- `incoming_ports[element_id]` records connected input ports.
- `outgoing_ports[element_id]` records connected output ports.
- Each element receives `configure_connected_output_ports(...)`.

During `validate()`:

- Every connection endpoint must reference an existing element.
- Source output ports are checked against static or dynamic output contracts.
- Target input ports are checked against static or dynamic input contracts.
- Format/depth compatibility is checked when both ports declare constraints.

Runtime packet compatibility is checked just before an element processes inputs.

## Scheduler Behavior

The scheduler is synchronous and frame-driven.

At a high level, `Pipeline.run()`:

1. Calls `start(context)` on every element.
2. Pulls packets from active sources.
3. Routes source outputs into per-port buffers.
4. Repeatedly processes non-source elements whose required input buffers are
   ready.
5. Stops when sources are exhausted and buffers are drained.
6. Calls `stop()` on every element in reverse order.

Buffers are keyed by `(target_element_id, target_port_name)`.

An element is ready when every required input port has at least one buffered
packet. Required input ports are:

- Connected static input ports, when present.
- Connected dynamic input ports matching declared prefixes.
- Otherwise, all static contract input ports.

For each ready element, the scheduler pops one packet from each required input
buffer, validates the input packets, calls `process(inputs)`, and routes returned
outputs.

If sources finish but buffered packets remain and no element can make progress,
the scheduler raises an error. This protects against mismatched stream lengths or
joiner compatibility problems.

## How Elements Interact

Elements interact only through `PacketInputs` and `PacketOutputs`.

```python
PacketInputs = dict[str, FramePacket | list[FramePacket]]
PacketOutputs = dict[str, list[FramePacket]]
```

In normal scheduler execution, each input port receives one `FramePacket` at a
time. Outputs are always lists so an element can emit zero, one, or more packets
on a port.

Common interaction patterns:

- Pass-through: return the same packet object on one or more output ports.
- Derivation: create a new NumPy array and a derived metadata object.
- Join: require multiple input ports and merge synchronized packets.
- Split: send the same packet or derived packets to multiple output ports.
- Sink: consume the packet and return an empty output dictionary.

When returning an output port, the port name must be declared by the element
contract as either a static output or a matching dynamic output.

## Metadata And Provenance Rules

Use these rules when implementing elements:

- If the frame data is unchanged and metadata is unchanged, pass through the
  same `FramePacket` object.
- If the frame data changes, create a new `FramePacket`.
- If width, height, channels, depth, format, index, timing, stream ID, or source
  ID changes, update metadata accordingly.
- Preserve useful `metadata.extra` values unless the element intentionally
  replaces them.
- Add element-specific audit values to `metadata.extra` when useful.
- Preserve parent provenance by using `metadata.derive(...)` or by explicitly
  building a `parents` tuple.

## CLI Parsing

The CLI parser converts a GStreamer-like expression into a `PipelineSpec`.

Basic shape:

```bash
uv run zpipe run "source-type name=s key=value ! transform-type name=t ! sink-type"
```

The parser supports:

- Linear chains separated by `!`.
- Named instances with `name=<id>`.
- Port references with `<element>.<port>`.
- Multi-line expressions.
- Comments in script files.
- `key=value` parameters.
- Automatic conversion of parameter values to bool, int, or float when possible.

The run command can read a pipeline expression from a file. Script files may use
placeholders such as `$1`, `$2`, and so on. Placeholders are replaced by CLI
arguments using shell-safe quoting.

```bash
uv run zpipe run --file pipeline.zpipe arg1 arg2
```

Use these introspection commands instead of documenting individual elements in
this file:

```bash
uv run zpipe list-elements
uv run zpipe list-elements --verbose
uv run zpipe describe <element>
```

## Adding A New Element

1. Choose the role.

   Place source elements in `src/sources/`, transform/control elements in
   `src/transformers/`, and sinks in `src/sinks/`.

2. Create a class extending `Source`, `Transformer`, or `Sink`.

   ```python
   from src.lib.elements import Transformer


   class MyTransform(Transformer):
       type_name = "my-transform"
   ```

3. Implement `contract()`.

   Include ports, parameters, description, optional subcategory, and compatibility
   rules. Use static ports for fixed interfaces and dynamic port prefixes for
   arbitrary numbered ports.

   ```python
   from src.lib.contracts import ElementContract, ParameterContract, PortContract


   @classmethod
   def contract(cls) -> ElementContract:
       return ElementContract(
           input_ports={"in": PortContract("in")},
           output_ports={"out": PortContract("out")},
           parameters={
               "amount": ParameterContract(
                   "amount",
                   "float",
                   default=1.0,
                   description="Processing strength.",
               ),
           },
           description="Apply a custom transformation.",
           subcategory="Custom",
       )
   ```

4. Validate parameters in `configure()`.

   Convert values to the expected Python types and raise `ValueError` for invalid
   combinations. Keep CLI-friendly aliases local to the element when needed.

5. Implement `process()`.

   ```python
   def process(self, inputs):
       packet = self._single_input(inputs)
       output_data = ...
       metadata = packet.metadata.derive(
           width=...,
           height=...,
           depth=...,
           channels=...,
           extra={
               **packet.metadata.extra,
               "processed_by": self.instance_id,
           },
       )
       return {"out": [FramePacket(data=output_data, metadata=metadata)]}
   ```

6. Use lifecycle hooks only when needed.

   Allocate external resources in `start()`, release them in `stop()`, and use
   `configure_connected_output_ports()` only for elements whose behavior depends
   on connected outputs.

7. Register the element.

   Add the import and class to `register_builtin_elements()` in
   `src/lib/registry.py`.

8. Add tests.

   Cover parameter validation, packet behavior, metadata behavior, port
   validation, parser smoke cases for unusual port/parameter syntax, registry
   visibility, and CLI introspection for registered elements.

## Development Notes

- The scheduler is synchronous; do not assume threaded execution.
- The graph is push-routed after source pulls; elements do not pull directly from
  upstream elements.
- Use contracts for structural validation and `process()` for data-dependent
  validation.
- Keep element implementation scoped: parameter parsing, validation, frame
  processing, metadata updates, and output construction should be easy to audit.
- Do not add element-specific usage documentation here. Keep this file focused on
  architecture and contribution mechanics.
