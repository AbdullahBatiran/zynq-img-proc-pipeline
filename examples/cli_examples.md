# CLI Examples

Run a linear processing pipeline:

```bash
zpipe run "filesrc path=input.mp4 ! resize width=640 height=480 ! hist_equalize mode=clahe ! filesink path=out.mp4"
```

Run two sources, resize both, combine them horizontally, and display the result:

```bash
zpipe run "
  filesrc name=a path=a.mp4 ! resize name=ra width=640 height=480
  filesrc name=b path=b.mp4 ! resize name=rb width=640 height=480
  ra.out ! combine.left name=c mode=horizontal
  rb.out ! c.right
  c.out ! displaysink window_name=combined fps=30
"
```

Display sinks sync to packet metadata FPS by default. Use `fps=30` to override
that rate, `wait_ms=33` for a fixed OpenCV delay, or `sync=false` to display as
fast as processing allows.
