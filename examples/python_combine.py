"""Run two file streams through resize and combine them for display."""

from src.lib.pipeline import ConnectionSpec, ElementSpec, Pipeline, PipelineSpec


spec = PipelineSpec(
    elements=[
        ElementSpec(id="src_a", type="filesrc", params={"path": "a.mp4"}),
        ElementSpec(id="src_b", type="filesrc", params={"path": "b.mp4"}),
        ElementSpec(id="resize_a", type="resize", params={"width": 640, "height": 480}),
        ElementSpec(id="resize_b", type="resize", params={"width": 640, "height": 480}),
        ElementSpec(id="combine", type="combine", params={"mode": "horizontal"}),
        ElementSpec(id="display", type="displaysink", params={"window_name": "combined"}),
    ],
    connections=[
        ConnectionSpec("src_a", "out", "resize_a", "in"),
        ConnectionSpec("src_b", "out", "resize_b", "in"),
        ConnectionSpec("resize_a", "out", "combine", "left"),
        ConnectionSpec("resize_b", "out", "combine", "right"),
        ConnectionSpec("combine", "out", "display", "in"),
    ],
)

Pipeline.from_spec(spec).run()
