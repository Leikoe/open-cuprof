# open-cuprof

Lightweight single-file header-only profiler for CUDA kernels with Chrome Trace / Perfetto export.

> **Note**: This project was created with AI assistance (Claude Code).

## Why open-cuprof?

- **Low overhead**: Minimal instrumentation using PTX special registers (`%laneid`, `%warpid`, `%ctaid`, `%globaltimer`)
- **Dead simple API**: Just `profiler.start_event("name")` and `profiler.end_event("name")`
- **No kernel signature changes**: Uses `__device__` globals - add profiling without modifying function signatures
- **Multiple profilers**: Run multiple independent profiler instances simultaneously
- **Per-warp profiling**: Track events independently for each warp
- **String literal section names**: Use readable names like `"mma"` directly in device code
- **Chrome Trace export**: View results in chrome://tracing or Perfetto UI
- **Single header-only file**: Just `#include "profiler.cuh"`

## Quick Start

```cpp
#include "profiler.cuh"

// Define profiler as __device__ global
__device__ WarpProfiler<512, 1> myprofiler;

__global__ void my_kernel() {
    // Profile from warp leader only
    if (profiler_is_warp_leader()) {
        myprofiler.start_event("compute");
        // ... work ...
        myprofiler.end_event("compute");
        
        myprofiler.start_event("sync");
        __syncthreads();
        myprofiler.end_event("sync");
    }
}

int main() {
    int num_blocks = 128;
    
    // Initialize profiler
    profiler_init(&myprofiler, num_blocks);
    
    // Run kernel (no signature changes needed!)
    my_kernel<<<num_blocks, 256>>>();
    
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
- **Chrome**: Navigate to `chrome://tracing` and load the file
- **Perfetto**: Open https://ui.perfetto.dev/ and load the file

## API Reference

### Device-side API

**Methods** (call from warp leader only):
- `profiler.start_event("section_name")` - Start recording a section
- `profiler.end_event("section_name")` - End recording a section (must match start)

**Helper**:
- `profiler_is_warp_leader()` - Returns true if current thread is the warp leader (lane 0)

String literals are automatically stored in device global memory and retrieved on the host.

### Host-side API

**Functions**:
- `profiler_init(&profiler, num_blocks)` - Initialize profiler and allocate device memory
- `profiler_export_and_cleanup(&profiler, "trace.json")` - Export to Chrome Trace JSON and free memory

### Template Parameters

```cpp
WarpProfiler<MAX_EVENTS, MAX_WARPS>
```

- `MAX_EVENTS`: Maximum events per warp per block (default: 128)
- `MAX_WARPS`: Maximum warps to track per block (default: 32)

Adjust these to match your profiling needs and available memory.

## Multiple Profilers

You can use multiple profilers simultaneously:

```cpp
__device__ WarpProfiler<256, 1> compute_profiler;
__device__ WarpProfiler<256, 1> memory_profiler;

profiler_init(&compute_profiler, num_blocks);
profiler_init(&memory_profiler, num_blocks);

// In kernel: use different profilers for different aspects
if (profiler_is_warp_leader()) {
    compute_profiler.start_event("gemm");
    // ...
    compute_profiler.end_event("gemm");
    
    memory_profiler.start_event("load");
    // ...
    memory_profiler.end_event("load");
}

profiler_export_and_cleanup(&compute_profiler, "compute.json");
profiler_export_and_cleanup(&memory_profiler, "memory.json");
```

## How It Works

1. **Device-side**: String literals in CUDA device code are compiled into device global memory with stable addresses
2. **Profiling**: The profiler stores pointers to these strings along with high-resolution timestamps from `%globaltimer`
3. **Export**: Strings are read byte-by-byte from device memory to avoid over-reading
4. **Output**: Results are formatted as Chrome Trace Event Format JSON with human-readable section names

The profiler uses PTX special registers for minimal overhead:
- `%laneid` - Lane ID within warp (0-31)
- `%warpid` - Warp ID within block
- `%ctaid.x` - Block ID in grid
- `%globaltimer` - High-resolution GPU timer

## Example Output

The Chrome Trace viewer will show:
- **Process (pid)**: Each CUDA block
- **Thread (tid)**: Each warp within the block
- **Events**: Duration bars with your custom section names

This makes it easy to visualize per-warp timing and identify bottlenecks.

## License

MIT
