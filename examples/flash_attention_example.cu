#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include "profiler.cuh"

/*
* Flash Attention v2 for Ampere (sm_80+)
 *
 * Key improvements over FA1:
 * - Parallelizes over sequence length dimension (better GPU utilization)
 * - Each warp handles multiple rows of Q
 * - Reduced shared memory usage through better partitioning
 * - Non-matmul FLOPs reduced (softmax computed in registers)
 *
 * Algorithm:
 * - Each block processes Br rows of Q
 * - Each warp within block processes subset of those rows
 * - Loop over K/V in chunks of Bc
 * - Maintains per-thread running statistics (m, l) for online softmax
 * - Uses registers for accumulation, shared memory for tiles
 */

// Tile sizes - tuned for Ampere
constexpr int Br = 64;  // Q rows per block
constexpr int Bc = 64;  // K/V columns per tile
constexpr int d = 64;   // Head dimension
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

// Define profiler
__device__ cuprof::Profiler<512, WARPS_PER_BLOCK> flash_profiler;

// Flash Attention v2 kernel
__global__ void flash_attention_v2_kernel(
    const float* Q,  // [N, d] - queries
    const float* K,  // [N, d] - keys
    const float* V,  // [N, d] - values
    float* O,        // [N, d] - output
    int N,           // sequence length
    float scale      // 1/sqrt(d)
) {
    __shared__ cuprof::BlockState block_state;
    block_state.init();

    // Block processes Br rows of Q
    int block_row_start = blockIdx.x * Br;

    // Warp-level indexing (FA2 parallelizes over warps)
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp handles Br/WARPS_PER_BLOCK rows
    int rows_per_warp = Br / WARPS_PER_BLOCK;
    int warp_row_start = block_row_start + warp_id * rows_per_warp;
    int warp_row_end = min(warp_row_start + rows_per_warp, N);

    // Shared memory for K, V tiles (Q is streamed from global memory)
    __shared__ float K_smem[Bc][d];
    __shared__ float V_smem[Bc][d];

    bool is_leader = cuprof::is_warp_leader();
    cuprof::Event total_event;

    if (is_leader) {
        total_event = flash_profiler.start("warp_total", &block_state);
    }

    // Per-thread registers for Q row (each thread handles one Q row)
    float Q_reg[d];
    float O_reg[d];
    float m_prev = -INFINITY;  // Running max
    float l_prev = 0.0f;       // Running sum

    // Load Q rows for this warp
    cuprof::Event load_q_event;
    if (is_leader) load_q_event = flash_profiler.start("load_Q", &block_state);

    int my_row = warp_row_start + lane_id;
    if (my_row < warp_row_end && lane_id < rows_per_warp) {
        #pragma unroll
        for (int i = 0; i < d; i++) {
            Q_reg[i] = Q[my_row * d + i];
            O_reg[i] = 0.0f;
        }
    }

    if (is_leader) flash_profiler.end(load_q_event);

    // Loop over K, V tiles
    int num_tiles = (N + Bc - 1) / Bc;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int kv_start = tile_idx * Bc;
        int kv_end = min(kv_start + Bc, N);
        int tile_size = kv_end - kv_start;

        cuprof::Event tile_event;
        if (is_leader) tile_event = flash_profiler.start("process_KV_tile", &block_state);

        // Cooperatively load K tile into shared memory
        cuprof::Event load_k_event;
        if (is_leader) load_k_event = flash_profiler.start("load_K_tile", &block_state);

        for (int i = threadIdx.x; i < Bc * d; i += THREADS_PER_BLOCK) {
            int row = i / d;
            int col = i % d;
            int global_row = kv_start + row;
            if (global_row < N && row < tile_size) {
                K_smem[row][col] = K[global_row * d + col];
            } else {
                K_smem[row][col] = 0.0f;
            }
        }
        __syncthreads();

        if (is_leader) flash_profiler.end(load_k_event);

        // Cooperatively load V tile into shared memory
        cuprof::Event load_v_event;
        if (is_leader) load_v_event = flash_profiler.start("load_V_tile", &block_state);

        for (int i = threadIdx.x; i < Bc * d; i += THREADS_PER_BLOCK) {
            int row = i / d;
            int col = i % d;
            int global_row = kv_start + row;
            if (global_row < N && row < tile_size) {
                V_smem[row][col] = V[global_row * d + col];
            } else {
                V_smem[row][col] = 0.0f;
            }
        }
        __syncthreads();

        if (is_leader) flash_profiler.end(load_v_event);

        // Compute attention scores S = Q @ K^T for this tile
        cuprof::Event qk_event;
        if (is_leader) qk_event = flash_profiler.start("compute_QK", &block_state);

        float S_reg[Bc];  // Attention scores in registers

        if (my_row < warp_row_end && lane_id < rows_per_warp) {
            #pragma unroll 4
            for (int j = 0; j < tile_size; j++) {
                float sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    sum += Q_reg[k] * K_smem[j][k];
                }
                S_reg[j] = sum * scale;
            }
            // Mask out invalid positions
            for (int j = tile_size; j < Bc; j++) {
                S_reg[j] = -INFINITY;
            }
        }

        if (is_leader) flash_profiler.end(qk_event);

        // Online softmax update (FA2: computed in registers, not shared memory)
        cuprof::Event softmax_event;
        if (is_leader) softmax_event = flash_profiler.start("online_softmax", &block_state);

        if (my_row < warp_row_end && lane_id < rows_per_warp) {
            // Find max in current tile
            float m_curr = -INFINITY;
            #pragma unroll 4
            for (int j = 0; j < tile_size; j++) {
                m_curr = fmaxf(m_curr, S_reg[j]);
            }

            // Update global max
            float m_new = fmaxf(m_prev, m_curr);

            // Compute exp and sum for current tile
            float l_curr = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < tile_size; j++) {
                S_reg[j] = expf(S_reg[j] - m_new);
                l_curr += S_reg[j];
            }

            // Rescale previous output and sum
            float scale_factor = expf(m_prev - m_new);
            #pragma unroll
            for (int i = 0; i < d; i++) {
                O_reg[i] *= scale_factor;
            }

            l_prev = l_prev * scale_factor + l_curr;
            m_prev = m_new;
        }

        if (is_leader) flash_profiler.end(softmax_event);

        // Accumulate O += P @ V (P are the normalized scores in S_reg)
        cuprof::Event pv_event;
        if (is_leader) pv_event = flash_profiler.start("accumulate_PV", &block_state);

        if (my_row < warp_row_end && lane_id < rows_per_warp) {
            #pragma unroll
            for (int i = 0; i < d; i++) {
                float sum = 0.0f;
                #pragma unroll 4
                for (int j = 0; j < tile_size; j++) {
                    sum += S_reg[j] * V_smem[j][i];
                }
                O_reg[i] += sum;
            }
        }

        if (is_leader) flash_profiler.end(pv_event);
        if (is_leader) flash_profiler.end(tile_event);

        __syncthreads();
    }

    // Final normalization and write output
    cuprof::Event write_event;
    if (is_leader) write_event = flash_profiler.start("write_output", &block_state);

    if (my_row < warp_row_end && lane_id < rows_per_warp) {
        #pragma unroll
        for (int i = 0; i < d; i++) {
            O[my_row * d + i] = O_reg[i] / l_prev;
        }
    }

    if (is_leader) flash_profiler.end(write_event);
    if (is_leader) flash_profiler.end(total_event);
}

// Reference attention implementation for correctness check
void reference_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    float scale
) {
    // Allocate temporary storage
    float* S = new float[N * N];

    // Compute S = Q @ K^T
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale;
        }
    }

    // Softmax per row
    for (int i = 0; i < N; i++) {
        // Find max
        float max_val = -INFINITY;
        for (int j = 0; j < N; j++) {
            max_val = fmaxf(max_val, S[i * N + j]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            S[i * N + j] = expf(S[i * N + j] - max_val);
            sum += S[i * N + j];
        }

        // Normalize
        for (int j = 0; j < N; j++) {
            S[i * N + j] /= sum;
        }
    }

    // Compute O = S @ V
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += S[i * N + j] * V[j * d + k];
            }
            O[i * d + k] = sum;
        }
    }

    delete[] S;
}

int main() {
    // Problem size
    const int N = 1024;  // Sequence length
    const float scale = 1.0f / sqrtf((float)d);

    printf("Flash Attention v2 Example (Ampere)\n");
    printf("  Sequence length: %d\n", N);
    printf("  Head dimension: %d\n", d);
    printf("  Tile sizes: Br=%d, Bc=%d\n", Br, Bc);
    printf("  Warps per block: %d\n", WARPS_PER_BLOCK);
    printf("  Rows per warp: %d\n\n", Br / WARPS_PER_BLOCK);

    // Allocate host memory
    float* h_Q = new float[N * d];
    float* h_K = new float[N * d];
    float* h_V = new float[N * d];
    float* h_O = new float[N * d];
    float* h_O_ref = new float[N * d];

    // Initialize with random data
    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        h_K[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        h_V[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, N * d * sizeof(float));
    cudaMalloc(&d_K, N * d * sizeof(float));
    cudaMalloc(&d_V, N * d * sizeof(float));
    cudaMalloc(&d_O, N * d * sizeof(float));

    // Copy to device
    cudaMemcpy(d_Q, h_Q, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, N * d * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    int num_blocks = (N + Br - 1) / Br;

    printf("Grid size: %d blocks, %d threads per block\n", num_blocks, THREADS_PER_BLOCK);
    printf("Total warps: %d\n\n", num_blocks * WARPS_PER_BLOCK);

    // Initialize profiler
    cuprof::init(&flash_profiler, num_blocks);

    // Warm-up run
    flash_attention_v2_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_Q, d_K, d_V, d_O, N, scale
    );
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    flash_attention_v2_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_Q, d_K, d_V, d_O, N, scale
    );
    cudaEventRecord(end);
    cudaDeviceSynchronize();

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, end);

    printf("Kernel execution time: %.3f ms\n", elapsed_ms);

    // Calculate throughput
    // FLOPs: 2 * N^2 * d (QK^T) + 2 * N^2 (softmax) + 2 * N^2 * d (PV)
    long long flops = 4LL * N * N * d + 2LL * N * N;
    double gflops = (flops / 1e9) / (elapsed_ms / 1000.0);
    printf("Performance: %.2f GFLOPS\n\n", gflops);

    // Export profiler trace
    cuprof::export_and_cleanup(&flash_profiler, "flash_attention_trace.json");
    printf("Profiler trace exported to flash_attention_trace.json\n\n");

    // Copy result back
    cudaMemcpy(h_O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute reference
    printf("Computing reference solution for correctness check...\n");
    reference_attention(h_Q, h_K, h_V, h_O_ref, N, scale);

    // Check correctness
    float max_error = 0.0f;
    float avg_error = 0.0f;
    for (int i = 0; i < N * d; i++) {
        float error = fabsf(h_O[i] - h_O_ref[i]);
        max_error = fmaxf(max_error, error);
        avg_error += error;
    }
    avg_error /= (N * d);

    printf("Max error: %.6e\n", max_error);
    printf("Avg error: %.6e\n", avg_error);

    bool passed = max_error < 1e-3f;
    printf("\nCorrectness check: %s\n", passed ? "PASSED" : "FAILED");

    if (!passed) {
        printf("\nShowing first few mismatches:\n");
        int count = 0;
        for (int i = 0; i < N && count < 10; i++) {
            for (int j = 0; j < d && count < 10; j++) {
                int idx = i * d + j;
                float error = fabsf(h_O[idx] - h_O_ref[idx]);
                if (error > 1e-3f) {
                    printf("  [%d,%d]: GPU=%.6f, CPU=%.6f, error=%.6e\n",
                           i, j, h_O[idx], h_O_ref[idx], error);
                    count++;
                }
            }
        }
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    delete[] h_O_ref;

    printf("\nView the trace at:\n");
    printf("  - Chrome: chrome://tracing\n");
    printf("  - Perfetto: https://ui.perfetto.dev/\n\n");
    printf("Flash Attention v2 profiling shows:\n");
    printf("  - Per-warp total execution time (better parallelism than FA1)\n");
    printf("  - Q loading (once per warp, streamed to registers)\n");
    printf("  - Per-tile K/V loading (cooperative across block)\n");
    printf("  - QK^T computation (in registers, parallel across warps)\n");
    printf("  - Online softmax (register-based, no shared memory)\n");
    printf("  - PV accumulation (register-based)\n");
    printf("  - Output writing\n\n");
    printf("Key FA2 improvements:\n");
    printf("  - Parallelism: Each warp independently processes %d rows\n", Br / WARPS_PER_BLOCK);
    printf("  - Memory: Reduced shared memory (only K/V tiles, no S matrix)\n");
    printf("  - Efficiency: Softmax computed in registers, not shared memory\n");

    return passed ? 0 : 1;
}
