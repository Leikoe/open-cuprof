# open-cuprof

Lightweight single-file header-only profiler for CUDA kernels with Chrome Trace / Perfetto export.

> **Note**: This project was created with AI assistance (Claude Code).

## Acknowledgments

This profiler was inspired by and borrows architectural ideas from [ThunderKittens' profiler](https://github.com/HazyResearch/ThunderKittens/blob/78c6c446afa3faafdca1d12cf874cc3662ef77e8/include/pyutils/profiler.cuh), particularly:
- Using PTX special registers for minimal overhead
- Per-warp event tracking without modifying kernel signatures
- Chrome Trace JSON export format

We extend these concepts with SM-based visualization, high-resolution `%clock64` timing, and per-SM timestamp normalization.

## Why open-cuprof?

- **Low overhead**: Minimal instrumentation using PTX special registers (`%laneid`, `%warpid`, `%ctaid`, `%smid`, `%clock64`)
- **Dead simple API**: Just `profiler.start_event("name")` and `profiler.end_event(event)`
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
__device__ cuprof::Profiler<512, 1> myprofiler;

__global__ void my_kernel() {
    // Profile from warp leader only
    if (cuprof::is_warp_leader()) {
        cuprof::Event compute_id = myprofiler.start_event("compute");
        // ... work ...
        myprofiler.end_event(compute_id);
        
        cuprof::Event sync_id = myprofiler.start_event("sync");
        __syncthreads();
        myprofiler.end_event(sync_id);
    }
}

int main() {
    int num_blocks = 128;
    
    // Initialize profiler
    cuprof::init(&myprofiler, num_blocks);
    
    // Run kernel (no signature changes needed!)
    my_kernel<<<num_blocks, 256>>>();
    
    // Export and cleanup
    cuprof::export_and_cleanup(&myprofiler, "trace.json");
}
```

## Building the Examples

```bash
make                           # Build all examples
./dgemm_example                # Tensor core DGEMM
./multi_warp_example           # Multiple warps with different workloads
./nested_events_example        # Nested event profiling
./flash_attention_example      # Flash Attention implementation
```

Each example generates a trace JSON file that you can view in:
- **Chrome**: Navigate to `chrome://tracing` and load the file
- **Perfetto**: Open https://ui.perfetto.dev/ and load the file

### Examples Overview

- **dgemm_example.cu**: Tensor core FP64 matrix multiplication using `mma.m8n8k4.f64`
- **multi_warp_example.cu**: Demonstrates per-warp profiling with 4 warps doing different computational workloads
- **nested_events_example.cu**: Shows hierarchical profiling with nested events (zero-overhead nesting)
- **flash_attention_example.cu**: Production-quality Flash Attention kernel with tiled computation and online softmax

## API Reference

### Device-side API

**Methods** (call from warp leader only):
- `cuprof::Event id = profiler.start_event("section_name")` - Start recording a section, returns event handle
- `profiler.end_event(id)` - End recording a section using the event handle

**Helper**:
- `cuprof::is_warp_leader()` - Returns true if current thread is the warp leader (lane 0)

**Types**:
- `cuprof::Event` - Event handle type. Default-constructed Events are invalid.
- `cuprof::Profiler<MAX_EVENTS, MAX_WARPS>` - Main profiler class

**Nesting**: Events can be nested by keeping track of event handles. The profiler has zero overhead for nesting - no loops or searches, just direct array indexing.

String literals are automatically stored in device global memory and retrieved on the host.

### Host-side API

**Functions**:
- `cuprof::init(&profiler, num_blocks)` - Initialize profiler and allocate device memory
- `cuprof::export_and_cleanup(&profiler, "trace.json")` - Export to Chrome Trace JSON and free memory

### Template Parameters

```cpp
cuprof::Profiler<MAX_EVENTS, MAX_WARPS>
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
2. **Profiling**: The profiler stores pointers to these strings along with high-resolution timestamps from `%clock64` (SM cycle counter)
3. **Export**: Strings are read byte-by-byte from device memory to avoid over-reading
4. **Output**: Results are formatted as Chrome Trace Event Format JSON with human-readable section names

The profiler uses PTX special registers for minimal overhead:
- `%laneid` - Lane ID within warp (0-31)
- `%warpid` - Warp ID within block
- `%ctaid.x` - Block ID in grid
- `%smid` - SM (Streaming Multiprocessor) identifier
- `%clock64` - High-resolution 64-bit SM cycle counter

By using `%clock64` (per-SM cycle counter) combined with `%smid` for grouping, the profiler achieves high resolution timing (~0.4 ns at 2.5 GHz) while avoiding synchronization issues between different SMs.

## Timing Assumptions and Limitations

The profiler makes the following assumptions for accurate timing:

1. **Constant clock rate**: `%clock64` runs at the GPU's base clock rate (queried via `cudaDevAttrClockRate`). GPU boost clocks or dynamic frequency scaling may affect accuracy.

2. **Per-SM monotonicity**: `%clock64` is assumed to be monotonically increasing within each SM during kernel execution. Clock discontinuities are filtered by the 1-second sanity check.

3. **Short kernel duration**: Events with durations > 1 second are assumed to be corrupted data and are clamped to 0. For very long-running kernels, consider alternative profiling methods.

4. **Proper event pairing**: Each `start_event("name")` must be followed by `end_event("name")` with the exact same name string. Mismatched pairs will not be recorded.

5. **Warp leader execution**: Profiling functions should only be called by the warp leader (lane 0). Use `profiler_is_warp_leader()` to check.

6. **No wraparound**: The 64-bit `%clock64` counter wraps after ~233 years at 2.5 GHz, so wraparound within a kernel is not a concern for normal use cases.

## Example Output

The Chrome Trace viewer will show:
- **Process (pid)**: Each SM (Streaming Multiprocessor) - shows actual hardware execution
- **Thread (tid)**: Each warp within the SM
- **Events**: Duration bars with your custom section names
- **Args**: Additional metadata including block ID and SM ID

This visualization makes it easy to:
- See actual hardware utilization across SMs
- Identify load imbalancing between SMs
- Visualize per-warp timing within each SM
- Understand how blocks are scheduled to hardware

## License

MIT
