# Python Stream Graph Interface Plan

## Summary
Build a Python video-stream pipeline framework with GStreamer-like concepts:
sources, transformers, sinks, ports, metadata, graph validation, and a CLI
expression layer. All frame movement uses `FramePacket`; raw NumPy frames do not
move through the pipeline without metadata.

## Implemented Layout
- Core framework: `src/lib/`
- Sources: `src/sources/`
- Sinks: `src/sinks/`
- Transformers: `src/transformers/`
- GUI-only transformer namespace: `src/transformers/gui/`
- Control-only transformer namespace: `src/transformers/control/`
- Python examples: `examples/python_linear.py`, `examples/python_combine.py`
- CLI examples: `examples/cli_examples.md`

## Implemented Elements
- `filesrc`: reads video frames with OpenCV and emits complete `FramePacket`
  metadata including width, height, fps, format, depth, channels, timestamp, and
  frame index.
- `resize`: resizes frames and updates dimension metadata.
- `hist_equalize`: applies global histogram equalization or CLAHE to 8-bit gray,
  BGR, or RGB frames.
- `combine`: combines two synchronized streams horizontally, vertically, or by
  overlay while preserving both input packet ids as provenance.
- `filesink`: writes frames to a video file.
- `displaysink`: displays frames with OpenCV.

## CLI Shape
Simple linear pipeline:

```bash
zpipe run "filesrc path=input.mp4 ! resize width=640 height=480 ! hist_equalize mode=clahe ! filesink path=out.mp4"
```

Named graph pipeline:

```bash
zpipe run "
  filesrc name=a path=a.mp4 ! resize name=ra width=640 height=480
  filesrc name=b path=b.mp4 ! resize name=rb width=640 height=480
  ra.out ! combine.left name=c mode=horizontal
  rb.out ! c.right
  c.out ! displaysink window_name=combined
"
```

## Next Steps
- Add more sources such as camera, RTSP, and synthetic test source.
- Add control transformers such as `tee`, frame limiter, frame dropper, and FPS
  controller.
- Add GUI transformers for debug overlays and live controls.
- Add richer CLI grammar after the Python graph API stabilizes.
