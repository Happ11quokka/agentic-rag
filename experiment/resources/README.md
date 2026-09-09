# Metal counter recording template

`metal-counters.tracetemplate` is an Instruments recording configuration derived
from Xcode 26.6 (17F113)'s built-in Metal System Trace template. Its GPU counter
profile is 13 (the APS counter profile supported by M3), with automatic GPU selection,
shader profiling disabled, and default GPU performance state. It does not force GPU
clocks or require runtime access to private profiling APIs.

The built-in Performance Limiters profile 3 returned an unsupported-profile error on
the development M3 Pro. Profile 13 exports `metal-gpu-counter-intervals` containing
GPU Read/Write Bandwidth (GB/s), ALU Utilization, F16/F32 and integer limiters, and
cache/MMU counters. The experiment validates those measurements after each bounded capture rather
than assuming profile numbers are portable across Xcode or GPU versions. The manifest records Xcode, macOS, profile ID, and template SHA-256.

References:

- [Apple: Measuring GPU memory bandwidth](https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth)
- [Apple: Reducing shader bottlenecks](https://developer.apple.com/documentation/xcode/reducing-shader-bottlenecks)
- [Apple: M3 profiling tools](https://developer.apple.com/videos/play/tech-talks/111374/)

Captures default to 2 seconds, with a maximum of 5 seconds per condition. The
runner stops both model servers at the deadline before waiting for finalization.
This avoids accumulating minutes of full-resolution counter data and frees model
memory before Instruments processes the trace.
