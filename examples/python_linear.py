"""Run a linear file -> resize -> equalize -> file pipeline."""

from src.lib.pipeline import ConnectionSpec, ElementSpec, Pipeline, PipelineSpec


spec = PipelineSpec(
    elements=[
        ElementSpec(id="src", type="filesrc", params={"path": "input.mp4"}),
        ElementSpec(id="resize", type="resize", params={"width": 640, "height": 480}),
        ElementSpec(id="eq", type="hist_equalize", params={"bins": 256}),
        ElementSpec(id="out", type="filesink", params={"path": "out.mp4"}),
    ],
    connections=[
        ConnectionSpec("src", "out", "resize", "in"),
        ConnectionSpec("resize", "out", "eq", "in"),
        ConnectionSpec("eq", "out", "out", "in"),
    ],
)

Pipeline.from_spec(spec).run()
