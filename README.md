# open-cuprof

Lightweight single-file header-only profiler for CUDA kernels with Chrome Trace / Perfetto export.

> **Note**: This project was created with AI assistance (Claude Code).

## Features

- **Per-warp profiling**: Track events independently for each warp
- **No kernel signature changes**: Uses `__device__` globals
- **Chrome Trace export**: View results in chrome://tracing or Perfetto UI
- **Single header-only file**: Just include `profiler.cuh`
- **Automatic warp/block detection**: Uses PTX special registers

## Quick Start

```cpp
#include "profiler.cuh"

// Define profiler as __device__ global
__device__ WarpProfiler<512, 1> myprofiler;

__global__ void my_kernel() {
    // Profile from warp leader only
    if ((threadIdx.x % 32) == 0) {
        myprofiler.start_event(0);  // section 0
        // ... work ...
        myprofiler.end_event(0);
    }
}

int main() {
    // Initialize profiler
    profiler_init(&myprofiler, num_blocks);
    
    // Run kernel
    my_kernel<<<blocks, threads>>>();
    
    // Export and cleanup
    profiler_export_and_cleanup(&myprofiler, "trace.json");
}
```

## Building the Example

```bash
make
./dgemm_example
```

This will generate a `trace.json` file that you can view in:
- Chrome: chrome://tracing
- Perfetto: https://ui.perfetto.dev/

## API

### Host-side (free functions)

- `profiler_init(&profiler, num_blocks)` - Initialize profiler with device memory
- `profiler_export_and_cleanup(&profiler, "trace.json")` - Export to JSON and free memory

### Device-side (methods)

- `profiler.start_event(section_id)` - Start recording a section (call from warp leader only)
- `profiler.end_event(section_id)` - End recording a section (call from warp leader only)

## Template Parameters

```cpp
WarpProfiler<MAX_EVENTS, MAX_WARPS>
```

- `MAX_EVENTS`: Maximum events per warp per block (default: 128)
- `MAX_WARPS`: Maximum warps to track per block (default: 32)

## License

MIT
