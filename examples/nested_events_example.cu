#include <cuda_runtime.h>
#include <stdio.h>
#include "profiler.cuh"

// Define the profiler as a __device__ global
__device__ WarpProfiler<512, 4> myprofiler;

// Kernel demonstrating nested event profiling
// This shows how to profile hierarchical operations:
// - Outer events that span multiple inner events
// - Loop iterations as nested scopes
// - Phases within each iteration
// Chrome Trace will display nested events stacked vertically
__global__ void nested_events_kernel(
    float *input,
    float *output,
    int n,
    int num_iterations
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    bool is_warp_leader = profiler_is_warp_leader();
    
    if (tid >= n) return;
    
    float value = input[tid];
    
    // Profile the entire computation (outer event)
    EventId total_id;
    if (is_warp_leader) total_id = myprofiler.start_event("total_computation");
    
    // Loop through iterations - each iteration is a nested scope
    for (int iter = 0; iter < num_iterations; iter++) {
        // Profile this iteration (nested within total_computation)
        EventId iter_id;
        if (is_warp_leader) iter_id = myprofiler.start_event("iteration");
        
        // Phase 1: Data preparation (nested within iteration)
        EventId prep_id;
        if (is_warp_leader) prep_id = myprofiler.start_event("prepare");
        float temp = value * 0.1f;
        if (is_warp_leader) myprofiler.end_event(prep_id);
        
        // Phase 2: Heavy computation (nested within iteration)
        EventId comp_id;
        if (is_warp_leader) comp_id = myprofiler.start_event("compute");
        #pragma unroll 1
        for (int i = 0; i < 20; i++) {
            temp = sqrtf(temp * temp + 1.0f);
        }
        if (is_warp_leader) myprofiler.end_event(comp_id);
        
        // Phase 3: Update (nested within iteration)
        EventId upd_id;
        if (is_warp_leader) upd_id = myprofiler.start_event("update");
        value += temp;
        if (is_warp_leader) myprofiler.end_event(upd_id);
        
        // End iteration event
        if (is_warp_leader) myprofiler.end_event(iter_id);
    }
    
    // Final reduction phase (nested within total_computation)
    EventId final_id;
    if (is_warp_leader) final_id = myprofiler.start_event("finalize");
    value = value / static_cast<float>(num_iterations);
    if (is_warp_leader) myprofiler.end_event(final_id);
    
    // End total_computation event
    if (is_warp_leader) myprofiler.end_event(total_id);
    
    // Store result
    if (tid < n) output[tid] = value;
}

int main() {
    const int N = 4096;
    const int NUM_ITERATIONS = 5;
    const int THREADS_PER_BLOCK = 128;
    const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    printf("Running nested events example:\n");
    printf("  Elements: %d\n", N);
    printf("  Iterations per thread: %d\n", NUM_ITERATIONS);
    printf("  Blocks: %d, Threads per block: %d\n\n", NUM_BLOCKS, THREADS_PER_BLOCK);
    
    // Initialize profiler
    profiler_init(&myprofiler, NUM_BLOCKS);
    
    // Allocate host memory
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    
    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i % 100) / 10.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    // Launch kernel
    cudaEventRecord(start);
    nested_events_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output, N, NUM_ITERATIONS);
    cudaEventRecord(end);
    cudaDeviceSynchronize();
    
    // Compute elapsed time
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    printf("Kernel execution time: %.3f ms\n", elapsed_ms);
    
    // Export profiler data
    profiler_export_and_cleanup(&myprofiler, "nested_events_trace.json");
    printf("Profiler trace exported to nested_events_trace.json\n\n");
    
    printf("View the trace at:\n");
    printf("  - Chrome: chrome://tracing\n");
    printf("  - Perfetto: https://ui.perfetto.dev/\n\n");
    
    printf("In the trace viewer, you should see:\n");
    printf("  - 'total_computation' spans the entire kernel\n");
    printf("  - %d 'iteration' events (one per iteration)\n", NUM_ITERATIONS);
    printf("  - Within each iteration: 'prepare', 'compute', 'update'\n");
    printf("  - 'finalize' at the end\n\n");
    
    printf("This demonstrates hierarchical profiling where:\n");
    printf("  - Outer events contain inner events\n");
    printf("  - You can see the breakdown of time within each iteration\n");
    printf("  - Chrome Trace will show nested events stacked vertically\n");
    
    // Copy result back and verify
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaEventDestroy(end);
    cudaEventDestroy(start);
    cudaFree(d_output);
    cudaFree(d_input);
    free(h_output);
    free(h_input);
    
    printf("\nExample completed successfully!\n");
    
    return 0;
}
