#include <cstdint>
#include <cuda_runtime.h>
#include <stdio.h>
#include "profiler.cuh"

// Define the profiler as a __device__ global
// We'll have up to 4 warps per block, with plenty of event space
__device__ cuprof::Profiler<256, 4> myprofiler;

// Simple kernel where each warp does different work
__global__ void multi_warp_kernel(
    float *input_a,
    float *input_b, 
    float *output,
    int n
) {
    myprofiler.block_init();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int warp_id;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp_id));
    
    bool is_warp_leader = cuprof::is_warp_leader();
    
    // Each warp performs different operations
    if (warp_id == 0) {
        // Warp 0: Vector addition
        cuprof::Event warp0_load_id;
        if (is_warp_leader) warp0_load_id = myprofiler.start("warp0_load");
        float a = (tid < n) ? input_a[tid] : 0.0f;
        float b = (tid < n) ? input_b[tid] : 0.0f;
        if (is_warp_leader) myprofiler.end(warp0_load_id);
        
        cuprof::Event warp0_add_id;
        if (is_warp_leader) warp0_add_id = myprofiler.start("warp0_add");
        float result = a + b;
        if (is_warp_leader) myprofiler.end(warp0_add_id);
        
        cuprof::Event warp0_store_id;
        if (is_warp_leader) warp0_store_id = myprofiler.start("warp0_store");
        if (tid < n) output[tid] = result;
        if (is_warp_leader) myprofiler.end(warp0_store_id);
        
    } else if (warp_id == 1) {
        // Warp 1: Vector multiplication with heavy compute loop
        cuprof::Event warp1_load_id;
        if (is_warp_leader) warp1_load_id = myprofiler.start("warp1_load");
        float a = (tid < n) ? input_a[tid] : 0.0f;
        float b = (tid < n) ? input_b[tid] : 0.0f;
        if (is_warp_leader) myprofiler.end(warp1_load_id);
        
        cuprof::Event warp1_multiply_id;
        if (is_warp_leader) warp1_multiply_id = myprofiler.start("warp1_multiply");
        float result = a * b;
        if (is_warp_leader) myprofiler.end(warp1_multiply_id);
        
        cuprof::Event warp1_heavy_compute_id;
        if (is_warp_leader) warp1_heavy_compute_id = myprofiler.start("warp1_heavy_compute");
        // Heavy computation loop to show measurable duration
        #pragma unroll 1
        for (int i = 0; i < 100; i++) {
            result = sqrtf(result * result + 1.0f);
        }
        if (is_warp_leader) myprofiler.end(warp1_heavy_compute_id);
        
        cuprof::Event warp1_store_id;
        if (is_warp_leader) warp1_store_id = myprofiler.start("warp1_store");
        if (tid < n) output[tid] = result;
        if (is_warp_leader) myprofiler.end(warp1_store_id);
        
    } else if (warp_id == 2) {
        // Warp 2: Iterative computation with sync
        cuprof::Event warp2_load_id;
        if (is_warp_leader) warp2_load_id = myprofiler.start("warp2_load");
        float a = (tid < n) ? input_a[tid] : 0.0f;
        float b = (tid < n) ? input_b[tid] : 0.0f;
        if (is_warp_leader) myprofiler.end(warp2_load_id);
        
        cuprof::Event warp2_compute_id;
        if (is_warp_leader) warp2_compute_id = myprofiler.start("warp2_compute");
        float result = a;
        #pragma unroll 1
        for (int i = 0; i < 50; i++) {
            result = expf(result * 0.01f) * b;
        }
        if (is_warp_leader) myprofiler.end(warp2_compute_id);
        
        cuprof::Event warp2_sync_id;
        if (is_warp_leader) warp2_sync_id = myprofiler.start("warp2_sync");
        __syncthreads();
        if (is_warp_leader) myprofiler.end(warp2_sync_id);
        
        cuprof::Event warp2_store_id;
        if (is_warp_leader) warp2_store_id = myprofiler.start("warp2_store");
        if (tid < n) output[tid] = result;
        if (is_warp_leader) myprofiler.end(warp2_store_id);
        
    } else if (warp_id == 3) {
        // Warp 3: Transcendental function heavy workload
        cuprof::Event warp3_load_id;
        if (is_warp_leader) warp3_load_id = myprofiler.start("warp3_load");
        float a = (tid < n) ? input_a[tid] : 0.0f;
        float b = (tid < n) ? input_b[tid] : 0.0f;
        if (is_warp_leader) myprofiler.end(warp3_load_id);
        
        cuprof::Event warp3_trig_compute_id;
        if (is_warp_leader) warp3_trig_compute_id = myprofiler.start("warp3_trig_compute");
        float result = 0.0f;
        #pragma unroll 1
        for (int i = 0; i < 30; i++) {
            result += sinf(a * 0.1f * i) * cosf(b * 0.1f * i);
        }
        if (is_warp_leader) myprofiler.end(warp3_trig_compute_id);
        
        cuprof::Event warp3_finalize_id;
        if (is_warp_leader) warp3_finalize_id = myprofiler.start("warp3_finalize");
        result = logf(fabsf(result) + 1.0f);
        if (is_warp_leader) myprofiler.end(warp3_finalize_id);
        
        cuprof::Event warp3_store_id;
        if (is_warp_leader) warp3_store_id = myprofiler.start("warp3_store");
        if (tid < n) output[tid] = result;
        if (is_warp_leader) myprofiler.end(warp3_store_id);
    }
}

int main() {
    const int N = 4096;  // Size of arrays
    const int THREADS_PER_BLOCK = 128;  // 4 warps per block (32 threads each)
    const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    printf("Running multi-warp example with %d blocks, %d threads per block\n", 
           NUM_BLOCKS, THREADS_PER_BLOCK);
    printf("Total warps: %d (4 per block)\n", NUM_BLOCKS * 4);
    
    // Initialize profiler
    cuprof::init(&myprofiler, NUM_BLOCKS);
    
    // Allocate host memory
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i / 100.0f;
        h_b[i] = (float)(N - i) / 100.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    // Launch kernel with timing
    cudaEventRecord(start);
    multi_warp_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_out, N);
    cudaEventRecord(end);
    cudaDeviceSynchronize();
    
    // Compute elapsed time
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    printf("Kernel execution time: %.3f ms\n", elapsed_ms);
    
    // Export profiler data
    cuprof::export_and_cleanup(&myprofiler, "multi_warp_trace.json");
    printf("Profiler trace exported to multi_warp_trace.json\n");
    printf("\nView the trace at:\n");
    printf("  - Chrome: chrome://tracing\n");
    printf("  - Perfetto: https://ui.perfetto.dev/\n");
    printf("\nYou should see 4 different warps per block, each doing different work!\n");
    
    // Copy result back (optional, just for validation)
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaEventDestroy(end);
    cudaEventDestroy(start);
    cudaFree(d_out);
    cudaFree(d_b);
    cudaFree(d_a);
    free(h_out);
    free(h_b);
    free(h_a);
    
    printf("\nExample completed successfully!\n");
    
    return 0;
}
