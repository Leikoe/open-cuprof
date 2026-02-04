#include <cstdint>
#include <cuda_runtime.h>
#include <stdio.h>
#include "profiler.cuh"

// mma.m8n8k4 .f64 TENSOR CORE INFORMATION, we use block size = TC size
#define BM 8
#define BN 8
#define BK 4
#define N_ACC_PER_THREAD 2

// Define profiler section names

// Define the profiler as a __device__ global
// Use smaller event count since we only have 4 sections per iteration
__device__ WarpProfiler<512, 1> myprofiler;  // 512 events, 1 warp (only profiling warp 0)

// TC DGEMM
// requires: A row major, B col major, C row major.
// Note: TNT is BLAS notation abuse
__global__ void dgemm_kernel_tnt(int M, int N, int K,
                                 double *A, // (M, K)
                                 double *B, // (K, N)
                                 double *C, // (M, N)
                                 int num_blocks_to_profile
) {
    int block_m = blockIdx.x;
    int block_n = blockIdx.y;
    
    // Calculate linear block ID for profiling bounds check
    int linear_block_id = blockIdx.y * gridDim.x + blockIdx.x;
    bool should_profile = linear_block_id < num_blocks_to_profile;

    // indexing formulas from: https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-884-a-f64
    int a_row = threadIdx.x / BK;
    int a_col = threadIdx.x % BK;
    // indexing formulas from: https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-884-b-f64
    int b_row = threadIdx.x % BK;
    int b_col = threadIdx.x / BK;

    double rA, rB, rC[N_ACC_PER_THREAD];

    // zero accumulators
    for (int i = 0; i < N_ACC_PER_THREAD; i++) {
        rC[i] = 0.0;
    }

    // Profile from warp leader (lane 0) and only if this block should be profiled
    bool is_warp_leader = (threadIdx.x % 32) == 0;
    bool do_profile = is_warp_leader && should_profile;

    for (int block_k = 0; block_k < K / BK; block_k++) {
        // rA load
        if (do_profile) myprofiler.start_event("load_A");
        {
            int m = block_m * BM + a_row;
            int k = block_k * BK + a_col;
            rA = A[m * K + k];
        }
        if (do_profile) myprofiler.end_event("load_A");
        
        // rB load
        if (do_profile) myprofiler.start_event("load_B");
        {
            int k = block_k * BK + b_row;
            int n = block_n * BN + b_col;
            rB = B[k * N + n];
        }
        if (do_profile) myprofiler.end_event("load_B");

        // MMA instruction
        if (do_profile) myprofiler.start_event("mma");
        // instr doc: https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%4, %5};"
                     : "=d"(rC[0]), "=d"(rC[1]) // outputs (the two accumulators)
                     : "d"(rA), "d"(rB), "d"(rC[0]), "d"(rC[1]) // inputs (a,b,c)
                     : "memory"); // hint that this instruction modifies memory
        if (do_profile) myprofiler.end_event("mma");
    }

    // Store results
    if (do_profile) myprofiler.start_event("store_C");
    // indexing formulas from: https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-884-c-f64
    {
        int groupID = threadIdx.x / BK;
        int threadID_in_group = threadIdx.x % BK;
        int row = groupID;
        for (int i = 0; i < N_ACC_PER_THREAD; i++) {
            int col = (threadID_in_group * 2) + (i & 0x1);
            C[(block_m * BM + row) * N + (block_n * BN + col)] = rC[i];
        }
    }
    if (do_profile) myprofiler.end_event("store_C");
}

int main() {
    // init problem size, here we want to show a single tensor core call, so we use its size
    int M = 1024;
    int N = 1024;
    int K = 1024;

    // Calculate grid size
    dim3 grid_size(M / BM, N / BN);
    
    // Only profile a subset of blocks to save memory (profile first 256 blocks)
    int num_blocks_to_profile = 256;
    
    // Initialize profiler with limited blocks
    profiler_init(&myprofiler, num_blocks_to_profile);

    // allocate host buffers
    double *a = (double *)malloc(sizeof(double) * M * K);
    double *b = (double *)malloc(sizeof(double) * K * N);
    double *c = (double *)malloc(sizeof(double) * M * N);

    // init inputs
    for (int i = 0; i < M * K; i++)
        a[i] = 2 * (rand() / double(RAND_MAX)) - 1;
    for (int i = 0; i < K * N; i++)
        b[i] = 2 * (rand() / double(RAND_MAX)) - 1;

    // allocate device buffers
    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeof(double) * M * K);
    cudaMalloc(&d_b, sizeof(double) * K * N);
    cudaMalloc(&d_c, sizeof(double) * M * N);

    // copy in inputs
    cudaMemcpy(d_a, a, sizeof(double) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(double) * K * N, cudaMemcpyHostToDevice);

    // create cuda events (for timing)
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // launch kernel & time execution
    dim3 block_size(32);
    cudaEventRecord(start); // record start
    dgemm_kernel_tnt<<<grid_size, block_size>>>(M, N, K, d_a, d_b, d_c, num_blocks_to_profile);
    cudaEventRecord(end); // record end
    cudaDeviceSynchronize(); // wait for completion of everything

    // compute elapsed time and achieved performance
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    double s = (double)elapsed_ms / 1000.0;
    double gflop = (2.0 * M * N * K) * 1e-9;
    printf("%f GFlops\n", gflop / s);

    // Export profiler data
    profiler_export_and_cleanup(&myprofiler, "trace.json");
    printf("Profiler trace exported to trace.json\n");

    // correctness check
    cudaMemcpy(c, d_c, sizeof(double) * M * N, cudaMemcpyDeviceToHost); // copy out result
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                acc += a[i * K + k] * b[k * N + j];
            }

            if (abs(acc - c[i * N + j]) > 0.001) {
                printf("BAD AT %d %d\n", i, j);
                exit(1);
            }
        }
    }

    printf("Correctness check passed!\n");

    // destroy cuda events
    cudaEventDestroy(end);
    cudaEventDestroy(start);
    // free device buffers
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
    // free host buffers
    free(c);
    free(b);
    free(a);

    return 0;
}
