/*
 * cuprof basic example
 *
 * Demonstrates usage of the cuprof profiler with a simple CUDA kernel.
 * After running, open the output trace.json in:
 *   - Chrome: chrome://tracing
 *   - Perfetto: https://ui.perfetto.dev
 */

#include "../cuprof.cuh"
#include <cstdio>

// Example kernel with profiled sections
__global__ void example_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Profile the initialization section
    CUPROF_START("init");
    float val = data[idx];
    CUPROF_END("init");

    // Profile computation section
    CUPROF_START("compute");
    for (int i = 0; i < 100; i++) {
        val = sinf(val) * cosf(val) + 0.1f;
    }
    CUPROF_END("compute");

    // Profile memory write
    CUPROF_START("store");
    data[idx] = val;
    CUPROF_END("store");
}

// Example with nested profiling
__global__ void nested_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    CUPROF_START("outer");

    float val = data[idx];

    CUPROF_START("inner_compute");
    for (int i = 0; i < 50; i++) {
        val = sqrtf(val * val + 1.0f);
    }
    CUPROF_END("inner_compute");

    data[idx] = val;

    CUPROF_END("outer");
}

// Example using CUPROF_SCOPE for automatic start/end
__global__ void scoped_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    {
        CUPROF_SCOPE("scoped_section");
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = expf(-val * 0.01f);
        }
        data[idx] = val;
    } // CUPROF_END automatically called here
}

int main() {
    printf("cuprof basic example\n");
    printf("====================\n\n");

    // Initialize profiler
    if (!cuprof::init()) {
        fprintf(stderr, "Failed to initialize profiler\n");
        return 1;
    }

    // Register section names for readable output
    cuprof::register_section("init");
    cuprof::register_section("compute");
    cuprof::register_section("store");
    cuprof::register_section("outer");
    cuprof::register_section("inner_compute");
    cuprof::register_section("scoped_section");

    // Allocate device memory
    const int N = 1024;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize data
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(i) * 0.01f;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    printf("Launching example_kernel...\n");
    example_kernel<<<grid, block>>>(d_data, N);

    printf("Launching nested_kernel...\n");
    nested_kernel<<<grid, block>>>(d_data, N);

    printf("Launching scoped_kernel...\n");
    scoped_kernel<<<grid, block>>>(d_data, N);

    cudaDeviceSynchronize();

    // Get and print statistics
    cuprof::Stats stats = cuprof::get_stats();
    printf("\nProfiling Statistics:\n");
    printf("  Total events:    %u\n", stats.total_events);
    printf("  Unique sections: %u\n", stats.unique_sections);
    printf("  Duration:        %.2f us\n", stats.duration_us);

    // Export trace
    cuprof::export_chrome_trace("trace.json");
    printf("\nOpen trace.json in Chrome (chrome://tracing) or Perfetto (https://ui.perfetto.dev)\n");

    // Cleanup
    cudaFree(d_data);
    delete[] h_data;
    cuprof::cleanup();

    return 0;
}
