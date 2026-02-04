#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include "profiler.cuh"

/*
 * Flash Attention v1 for Ampere (sm_80+)
 * 
 * Implements online softmax attention with tiling to reduce HBM accesses.
 * 
 * Algorithm:
 * - Processes Q in blocks, K and V in tiles
 * - Maintains running max and sum for numerically stable softmax
 * - Uses shared memory for tile storage
 * - Minimizes HBM reads/writes
 * 
 * This implementation uses:
 * - Warp-level matrix operations (wmma for FP16)
 * - Shared memory tiling
 * - Online softmax algorithm
 */

// Tile sizes - tuned for Ampere shared memory limits
constexpr int Br = 32;  // Q rows per block
constexpr int Bc = 32;  // K/V columns per tile
constexpr int d = 64;   // Head dimension

// Define profiler
__device__ cuprof::Profiler<512, 32> flash_profiler;

// FP32 Flash Attention kernel for Ampere
__global__ void flash_attention_kernel(
    const float* Q,  // [N, d] - queries
    const float* K,  // [N, d] - keys
    const float* V,  // [N, d] - values
    float* O,        // [N, d] - output
    int N,           // sequence length
    float scale      // 1/sqrt(d)
) {
    // Block handles Br rows of Q
    int block_row = blockIdx.x;
    int row_start = block_row * Br;
    int row_end = min(row_start + Br, N);
    
    // Thread indexing
    int tid = threadIdx.x;
    
    // Shared memory for tiles
    __shared__ float Q_smem[Br][d];
    __shared__ float K_smem[Bc][d];
    __shared__ float V_smem[Bc][d];
    __shared__ float S_smem[Br][Bc];  // Attention scores
    
    // Per-thread accumulators
    float O_local[d];
    float m_i = -INFINITY;  // Running max
    float l_i = 0.0f;       // Running sum
    
    bool is_leader = cuprof::is_warp_leader();
    cuprof::Event total_event;
    
    if (is_leader) {
        total_event = flash_profiler.start_event("total_attention");
    }
    
    // Load Q tile into shared memory
    cuprof::Event load_q_event;
    if (is_leader) load_q_event = flash_profiler.start_event("load_Q_tile");
    
    for (int i = tid; i < Br * d; i += blockDim.x) {
        int row = i / d;
        int col = i % d;
        int global_row = row_start + row;
        if (global_row < N) {
            Q_smem[row][col] = Q[global_row * d + col];
        } else {
            Q_smem[row][col] = 0.0f;
        }
    }
    __syncthreads();
    
    if (is_leader) flash_profiler.end_event(load_q_event);
    
    // Initialize output accumulator
    cuprof::Event init_event;
    if (is_leader) init_event = flash_profiler.start_event("init_accumulators");
    
    for (int i = 0; i < d; i++) {
        O_local[i] = 0.0f;
    }
    
    if (is_leader) flash_profiler.end_event(init_event);
    
    // Process K, V in tiles
    int num_tiles = (N + Bc - 1) / Bc;
    
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int col_start = tile_idx * Bc;
        int col_end = min(col_start + Bc, N);
        int tile_size = col_end - col_start;
        
        cuprof::Event tile_event;
        if (is_leader) tile_event = flash_profiler.start_event("process_tile");
        
        // Load K tile
        cuprof::Event load_k_event;
        if (is_leader) load_k_event = flash_profiler.start_event("load_K_tile");
        
        for (int i = tid; i < Bc * d; i += blockDim.x) {
            int row = i / d;
            int col = i % d;
            int global_row = col_start + row;
            if (global_row < N && row < tile_size) {
                K_smem[row][col] = K[global_row * d + col];
            } else {
                K_smem[row][col] = 0.0f;
            }
        }
        __syncthreads();
        
        if (is_leader) flash_profiler.end_event(load_k_event);
        
        // Load V tile
        cuprof::Event load_v_event;
        if (is_leader) load_v_event = flash_profiler.start_event("load_V_tile");
        
        for (int i = tid; i < Bc * d; i += blockDim.x) {
            int row = i / d;
            int col = i % d;
            int global_row = col_start + row;
            if (global_row < N && row < tile_size) {
                V_smem[row][col] = V[global_row * d + col];
            } else {
                V_smem[row][col] = 0.0f;
            }
        }
        __syncthreads();
        
        if (is_leader) flash_profiler.end_event(load_v_event);
        
        // Compute S = Q @ K^T (attention scores)
        cuprof::Event matmul_qk_event;
        if (is_leader) matmul_qk_event = flash_profiler.start_event("matmul_QK");
        
        for (int i = tid; i < Br * Bc; i += blockDim.x) {
            int row = i / Bc;
            int col = i % Bc;
            
            if (row_start + row < N && col < tile_size) {
                float sum = 0.0f;
                for (int k = 0; k < d; k++) {
                    sum += Q_smem[row][k] * K_smem[col][k];
                }
                S_smem[row][col] = sum * scale;
            } else {
                S_smem[row][col] = -INFINITY;
            }
        }
        __syncthreads();
        
        if (is_leader) flash_profiler.end_event(matmul_qk_event);
        
        // Online softmax update
        cuprof::Event softmax_event;
        if (is_leader) softmax_event = flash_profiler.start_event("online_softmax");
        
        // Each thread processes one row of Q
        if (tid < Br && row_start + tid < N) {
            int row = tid;
            
            // Find new max
            float m_i_new = m_i;
            for (int j = 0; j < tile_size; j++) {
                m_i_new = fmaxf(m_i_new, S_smem[row][j]);
            }
            
            // Compute exponentials and new sum
            float l_i_new = 0.0f;
            for (int j = 0; j < tile_size; j++) {
                S_smem[row][j] = expf(S_smem[row][j] - m_i_new);
                l_i_new += S_smem[row][j];
            }
            
            // Rescale previous output
            float scale_old = expf(m_i - m_i_new);
            for (int k = 0; k < d; k++) {
                O_local[k] = O_local[k] * scale_old;
            }
            
            // Update running statistics
            l_i = l_i * scale_old + l_i_new;
            m_i = m_i_new;
        }
        __syncthreads();
        
        if (is_leader) flash_profiler.end_event(softmax_event);
        
        // Accumulate O += P @ V
        cuprof::Event matmul_pv_event;
        if (is_leader) matmul_pv_event = flash_profiler.start_event("matmul_PV");
        
        if (tid < Br && row_start + tid < N) {
            int row = tid;
            for (int k = 0; k < d; k++) {
                float sum = 0.0f;
                for (int j = 0; j < tile_size; j++) {
                    sum += S_smem[row][j] * V_smem[j][k];
                }
                O_local[k] += sum;
            }
        }
        __syncthreads();
        
        if (is_leader) flash_profiler.end_event(matmul_pv_event);
        if (is_leader) flash_profiler.end_event(tile_event);
    }
    
    // Final normalization and write output
    cuprof::Event finalize_event;
    if (is_leader) finalize_event = flash_profiler.start_event("finalize_output");
    
    if (tid < Br && row_start + tid < N) {
        int row = tid;
        int global_row = row_start + row;
        for (int k = 0; k < d; k++) {
            O[global_row * d + k] = O_local[k] / l_i;
        }
    }
    
    if (is_leader) flash_profiler.end_event(finalize_event);
    if (is_leader) flash_profiler.end_event(total_event);
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
    const int N = 512;  // Sequence length
    const float scale = 1.0f / sqrtf((float)d);
    
    printf("Flash Attention Example (Ampere)\n");
    printf("  Sequence length: %d\n", N);
    printf("  Head dimension: %d\n", d);
    printf("  Tile sizes: Br=%d, Bc=%d\n", Br, Bc);
    printf("\n");
    
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
    int threads_per_block = 256;  // Multiple warps for parallelism
    
    printf("Grid size: %d blocks, %d threads per block\n", num_blocks, threads_per_block);
    printf("Total warps: %d\n\n", num_blocks * threads_per_block / 32);
    
    // Initialize profiler
    cuprof::init(&flash_profiler, num_blocks);
    
    // Warm-up run
    flash_attention_kernel<<<num_blocks, threads_per_block>>>(
        d_Q, d_K, d_V, d_O, N, scale
    );
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaEventRecord(start);
    flash_attention_kernel<<<num_blocks, threads_per_block>>>(
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
    printf("In the trace, you'll see:\n");
    printf("  - Total attention computation per block\n");
    printf("  - Per-tile processing (load K/V, matmul QK, softmax, matmul PV)\n");
    printf("  - Load operations for Q, K, V tiles\n");
    printf("  - Matrix multiplications (QK^T and PV)\n");
    printf("  - Online softmax updates\n");
    printf("  - Output finalization\n");
    
    return passed ? 0 : 1;
}
