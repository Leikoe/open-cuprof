/*
 * cuprof warp-specialized kernel example
 *
 * Demonstrates profiling warp-specialized kernels where different warps
 * perform different roles (e.g., producer/consumer pattern).
 * Only warp leaders record profiling events to reduce overhead.
 */

#include "../cuprof.cuh"
#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Simulated warp-specialized kernel with producer/consumer pattern
// Warp 0: Producer - generates data
// Warp 1+: Consumers - process data
__global__ void producer_consumer_kernel(float* shared_buffer, float* output, int n) {
    __shared__ float smem[256];
    __shared__ int ready_flag;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int block_offset = blockIdx.x * blockDim.x;

    if (threadIdx.x == 0) ready_flag = 0;
    __syncthreads();

    if (warp_id == 0) {
        // Producer warp
        CUPROF_WARP_START("producer_load");

        // Simulate loading data from global memory
        for (int i = lane_id; i < 256; i += 32) {
            int idx = block_offset + i;
            smem[i] = (idx < n) ? shared_buffer[idx] : 0.0f;
        }

        CUPROF_WARP_END("producer_load");

        __threadfence_block();
        if (lane_id == 0) {
            atomicExch(&ready_flag, 1);
        }

        CUPROF_WARP_START("producer_compute");

        // Producer does some additional work
        float sum = 0.0f;
        for (int i = lane_id; i < 256; i += 32) {
            sum += smem[i];
        }
        // Warp reduce
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        CUPROF_WARP_END("producer_compute");

    } else {
        // Consumer warps
        CUPROF_WARP_START("consumer_wait");

        // Wait for producer
        if (lane_id == 0) {
            while (atomicAdd(&ready_flag, 0) == 0) {
                // Spin wait
            }
        }
        __syncwarp();

        CUPROF_WARP_END("consumer_wait");

        CUPROF_WARP_START("consumer_compute");

        // Process data
        int consumer_id = warp_id - 1;
        int items_per_consumer = 256 / (blockDim.x / 32 - 1);
        int start = consumer_id * items_per_consumer;

        float result = 0.0f;
        for (int i = lane_id; i < items_per_consumer && (start + i) < 256; i += 32) {
            float val = smem[start + i];
            result += sinf(val) * cosf(val);
        }

        // Warp reduce
        for (int offset = 16; offset > 0; offset /= 2) {
            result += __shfl_down_sync(0xffffffff, result, offset);
        }

        // Write output
        if (lane_id == 0 && (block_offset + warp_id) < n) {
            output[block_offset / 32 + consumer_id] = result;
        }

        CUPROF_WARP_END("consumer_compute");
    }
}

// Warp-specialized kernel with explicit warp role IDs
__global__ void multi_role_kernel(float* data, int n) {
    int warp_in_block = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    // Assign roles based on warp ID
    // Role 0: Memory prefetch
    // Role 1: Computation
    // Role 2: Reduction
    int role = warp_in_block % 3;

    switch (role) {
        case 0:
            CUPROF_WARP_START_ID("prefetch", warp_in_block);
            // Simulate memory prefetch work
            for (volatile int i = 0; i < 100; i++) { }
            CUPROF_WARP_END_ID("prefetch", warp_in_block);
            break;

        case 1:
            CUPROF_WARP_START_ID("compute", warp_in_block);
            // Simulate compute work
            for (volatile int i = 0; i < 200; i++) { }
            CUPROF_WARP_END_ID("compute", warp_in_block);
            break;

        case 2:
            CUPROF_WARP_START_ID("reduce", warp_in_block);
            // Simulate reduction work
            for (volatile int i = 0; i < 150; i++) { }
            CUPROF_WARP_END_ID("reduce", warp_in_block);
            break;
    }
}

// Example using CUPROF_WARP_SCOPE for automatic cleanup
__global__ void scoped_warp_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    {
        CUPROF_WARP_SCOPE("warp_work");

        if (idx < n) {
            float val = data[idx];
            for (int i = 0; i < 50; i++) {
                val = sqrtf(val * val + 1.0f);
            }
            data[idx] = val;
        }
    } // End event automatically recorded by warp leader
}

int main() {
    printf("cuprof warp-specialized kernel example\n");
    printf("======================================\n\n");

    // Initialize profiler
    if (!cuprof::init()) {
        fprintf(stderr, "Failed to initialize profiler\n");
        return 1;
    }

    // Register section names
    cuprof::register_section("producer_load");
    cuprof::register_section("producer_compute");
    cuprof::register_section("consumer_wait");
    cuprof::register_section("consumer_compute");
    cuprof::register_section("prefetch");
    cuprof::register_section("compute");
    cuprof::register_section("reduce");
    cuprof::register_section("warp_work");

    // Allocate memory
    const int N = 4096;
    float *d_buffer, *d_output, *d_data;
    cudaMalloc(&d_buffer, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize data
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(i) * 0.01f + 1.0f;
    }
    cudaMemcpy(d_buffer, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch producer-consumer kernel
    // 4 warps per block: 1 producer + 3 consumers
    printf("Launching producer_consumer_kernel...\n");
    dim3 block1(128);  // 4 warps
    dim3 grid1((N + block1.x - 1) / block1.x);
    producer_consumer_kernel<<<grid1, block1>>>(d_buffer, d_output, N);

    // Launch multi-role kernel
    printf("Launching multi_role_kernel...\n");
    dim3 block2(192);  // 6 warps (2 of each role)
    dim3 grid2(4);
    multi_role_kernel<<<grid2, block2>>>(d_data, N);

    // Launch scoped warp kernel
    printf("Launching scoped_warp_kernel...\n");
    dim3 block3(256);
    dim3 grid3((N + block3.x - 1) / block3.x);
    scoped_warp_kernel<<<grid3, block3>>>(d_data, N);

    cudaDeviceSynchronize();

    // Get statistics
    cuprof::Stats stats = cuprof::get_stats();
    printf("\nProfiling Statistics:\n");
    printf("  Total events:    %u\n", stats.total_events);
    printf("  Unique sections: %u\n", stats.unique_sections);
    printf("  Duration:        %.2f us\n", stats.duration_us);

    // Export trace
    cuprof::export_chrome_trace("warp_trace.json");
    printf("\nOpen warp_trace.json in Chrome (chrome://tracing) or Perfetto (https://ui.perfetto.dev)\n");

    // Cleanup
    cudaFree(d_buffer);
    cudaFree(d_output);
    cudaFree(d_data);
    delete[] h_data;
    cuprof::cleanup();

    return 0;
}
