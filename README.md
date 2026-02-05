# open-cuprof
Lightweight single-file header-only profiler for CUDA kernels with Chrome Trace / Perfetto export.

> **Note**: Coding agents were used heavily for this project (Claude Code).

## Motivations
I wanted a simple profiler I could drop into any CUDA project and start profiling.

This header-only library is meant as a very light (low overhead) and non intrusive (doesn't require modifying kernel signature) way of vizualizing how the GPU executes a given kernel.

The API is as concise as I thought possible at the time of writing it. It also accepts actual strings for event names which other cuda profilers don't really and I felt like it was important for the developper experience.


## Quick Start
Include the library:
```cpp
#include "profiler.cuh"
```

Declare a global profiler instance:
```cpp
__device__ cuprof::Profiler<512, 1> myprofiler;
```

Record events from inside your kernel:
 **IMPORTANT:** only the leader of each warp can record events
```cpp
__global__ void my_kernel() {
    if (cuprof::is_warp_leader()) {
        cuprof::Event compute_id = myprofiler.start_event("compute");
    }    
    // ... work ...
    if (cuprof::is_warp_leader()) {
        myprofiler.end_event(compute_id);
    }
}
```

Now on the host side, init the profiler before your kernel:
```cpp
cuprof::init(&myprofiler, num_blocks);
```

And export the trace after:
```cpp
cuprof::export_and_cleanup(&myprofiler, "trace.json");
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


## Acknowledgments
This profiler was inspired by and borrows architectural ideas from [ThunderKittens' profiler](https://github.com/HazyResearch/ThunderKittens/blob/78c6c446afa3faafdca1d12cf874cc3662ef77e8/include/pyutils/profiler.cuh) and [Gau-nernst's profiler](https://github.com/gau-nernst/learn-cuda/blob/main/02e_matmul_sm100/profiler.h).
