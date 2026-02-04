# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**open-cuprof** is a lightweight, single-file, header-only profiler for CUDA kernels with Chrome Trace/Perfetto export. The profiler uses PTX special registers (`%laneid`, `%warpid`, `%ctaid`, `%smid`, `%clock64`) for minimal overhead and supports per-warp event tracking without requiring kernel signature changes.

## Core Design Principles

**CRITICAL - Read this before making any changes:**

1. **Minimal Overhead is Paramount**: This profiler is designed for high-performance kernels where every cycle counts. Any change that adds overhead (branches, loops, memory accesses) must be carefully justified. When in doubt, prefer simplicity over features.

2. **Simplest Possible Code**: The implementation should be as straightforward as possible. Avoid clever optimizations, complex data structures, or abstraction layers. Simple, readable code is easier to verify for correctness and performance.

3. **No Loops in Hot Path**: Device-side functions (`start_event()`, `end_event()`) must not contain loops or searches. These functions are called frequently within performance-critical kernels.

4. **Verify Generated Code**: When making changes to device-side code, always check the generated PTX/SASS to ensure no unexpected overhead is introduced.

## Build Commands

Build all examples:
```bash
make
```

Build individual examples:
```bash
make dgemm_example
make multi_warp_example
```

Clean build artifacts and generated traces:
```bash
make clean
```

Run examples:
```bash
./dgemm_example              # generates trace.json
./multi_warp_example         # generates multi_warp_trace.json
```

**Note**: The Makefile uses `-arch=sm_89` (Ada Lovelace architecture). Adjust this flag if targeting a different GPU architecture.

## Architecture

### Core Components

**profiler.cuh** (single header-only file):
- Namespace: `cuprof` - All types and functions are in this namespace
- `cuprof::Profiler<MAX_EVENTS, MAX_WARPS>` template struct: Main profiler implementation
  - `MAX_EVENTS`: Maximum events per warp per block (default: 128)
  - `MAX_WARPS`: Maximum warps to track per block (default: 32)
- Types:
  - `cuprof::Event` - Event handle returned by `start_event()` and passed to `end_event()`
  - `cuprof::EventData` - Internal struct for storing event data
- Device-side API: `start_event()`, `end_event()` called from warp leaders only
- Host-side API: `cuprof::init()`, `cuprof::export_and_cleanup()`
- Helper: `cuprof::is_warp_leader()` - PTX-based check for lane 0

### Key Design Patterns

**Device Global Profilers**: Profilers are declared as `__device__` globals to avoid modifying kernel signatures:
```cpp
__device__ cuprof::Profiler<512, 1> myprofiler;
```

**Per-Block Storage**: Each profiler maintains a `block_data` pointer to device memory allocated for all blocks. Device code uses `%ctaid.x` to index into per-block data.

**String Literal Handling**: Section names are passed as `const char*` pointers to device constant strings. Export code reads these byte-by-byte from device memory to avoid over-reading.

**Event Handle Pattern**: `start_event()` returns a `cuprof::Event` handle that must be passed to `end_event()`. This enables zero-overhead nested events through direct array indexing:
```cpp
if (cuprof::is_warp_leader()) {
    cuprof::Event event = myprofiler.start_event("section_name");
    // ... work ...
    myprofiler.end_event(event);
}
```

**Warp Leader Pattern**: Only the warp leader (lane 0) should call profiler methods to avoid redundant work.

**Chrome Trace Format**: Export uses the Trace Event Format with:
- `pid` (process ID) = SM ID (shows actual hardware execution)
- `tid` (thread ID) = warp ID within SM
- `args` = Additional metadata (block ID, SM ID)
- Timestamps normalized relative to global minimum and converted to microseconds

This visualization approach shows actual hardware utilization rather than logical block layout, making it easier to identify SM load imbalancing and understand how the CUDA scheduler maps blocks to physical hardware.

### Memory Management

1. Host calls `cuprof::init(&profiler, num_blocks)`:
   - Allocates device memory for `num_blocks` worth of `cuprof::Profiler` instances
   - Uses `cudaMemcpyToSymbol` to copy profiler struct to `__device__` global

2. Device code directly indexes into `block_data` array using block/warp IDs

3. Host calls `cuprof::export_and_cleanup(&profiler, "trace.json")`:
   - Uses `cudaMemcpyFromSymbol` to retrieve profiler struct
   - Copies all per-block data from device to host
   - Retrieves string literals byte-by-byte from device memory
   - Exports Chrome Trace JSON
   - Frees all device memory

## Example Structure

**examples/dgemm_example.cu**: Demonstrates profiling a tensor core DGEMM kernel using `mma.m8n8k4.f64` PTX instruction with profiling of:
- `load_A`: Loading A matrix elements
- `load_B`: Loading B matrix elements  
- `mma`: Tensor core matrix multiply-accumulate
- `store_C`: Storing result to C matrix

**examples/multi_warp_example.cu**: Demonstrates per-warp profiling with multiple warps per block (4 warps, 128 threads total). Each warp performs different computational workloads with measurable timing:
- Warp 0: Simple vector addition (load → add → store)
- Warp 1: Vector multiplication followed by heavy sqrt loop (100 iterations)
- Warp 2: Exponential function loop (50 iterations) with `__syncthreads()` 
- Warp 3: Trigonometric function loop (30 sin/cos iterations) with log finalization

This example shows how the profiler tracks independent work across warps within the same block. The different workloads produce measurable timing differences, making it easy to compare warp performance. When viewed in Chrome Trace, each warp appears as a separate thread (tid) within each block (pid), with clearly visible duration bars showing the relative cost of different operations.

**examples/nested_events_example.cu**: Demonstrates hierarchical profiling with nested events. Shows zero-overhead nesting using event handles - no loops or searches needed. Features:
- Outer event spanning entire computation
- Inner events for each iteration
- Further nested events within iterations (prepare, compute, update)
- Direct array indexing for O(1) event access

**examples/flash_attention_example.cu**: Production-quality Flash Attention v1 implementation for Ampere with comprehensive profiling. Demonstrates profiling of a real-world optimized kernel with:
- Total attention computation timing
- Per-tile processing (16 tiles for 512 sequence length)
- Individual operations: load Q/K/V tiles, matmul QK^T, online softmax, matmul PV
- Output finalization
- Tiled computation to minimize HBM accesses
- Online softmax algorithm for numerical stability
- Shared memory management (32x32 tiles to fit within 48KB limit)

This example shows how the profiler scales to complex kernels with multiple nested operations and demonstrates the overhead is negligible even with extensive instrumentation.

All examples show profiling without kernel signature changes, using only warp leader checks.

## Modifying the Profiler

When adding new features to `profiler.cuh`:

- **Device-side methods**: Must use PTX inline assembly for special registers (`%laneid`, `%warpid`, `%ctaid.x`, `%smid`, `%clock64`)
- **Event overflow**: Both `start_event()` and `end_event()` include bounds checking (`if (idx >= MAX_EVENTS) return`)
- **Event validity**: Events are only marked valid (`.valid = 1`) when `end_event()` is called with matching section name
- **Timestamp conversion**: `%clock64` cycle counts are converted to microseconds using GPU clock rate, normalized per-SM to handle per-SM cycle counters
- **Export format**: Chrome Trace uses "X" phase (complete events) with `ts` (timestamp) and `dur` (duration) in microseconds
- **Per-SM normalization**: Since `%clock64` is a per-SM counter, timestamps are normalized to each SM's minimum time to ensure proper visualization

## Viewing Traces

Generated `trace.json` files can be viewed in:
- Chrome: `chrome://tracing`
- Perfetto: https://ui.perfetto.dev/
