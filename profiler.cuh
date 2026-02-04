#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

// Helper to check if current thread is warp leader
__device__ __forceinline__ bool profiler_is_warp_leader() {
    return (threadIdx.x % 32) == 0;
}


/**
 * @brief Lightweight per-warp profiler for CUDA kernels with Chrome Trace/Perfetto export.
 *
 * Usage:
 *
 * 1. Define section names at global scope:
 *
 *      PROFILER_DEFINE_SECTION(mma);
 *      PROFILER_DEFINE_SECTION(load_A);
 *      PROFILER_DEFINE_SECTION(load_B);
 *
 * 2. Define a __device__ global profiler instance:
 *
 *      __device__ WarpProfiler<> myprofiler;
 *
 * 3. Host side - initialize:
 *
 *      profiler_init(&myprofiler, num_blocks);
 *
 * 4. Device side - record events from warp leader only:
 *
 *      if (warp_leader_condition) {
 *          myprofiler.start_event("mma");
 *          // ... work ...
 *          myprofiler.end_event("mma");
 *      }
 *
 * 5. Host side - export and cleanup:
 *
 *      profiler_export_and_cleanup(&myprofiler, "trace.json");
 */
template <int MAX_EVENTS = 128, int MAX_WARPS = 32>
struct __align__(16) WarpProfiler {
    struct Event {
        uint64_t start_time;
        uint64_t end_time;
        const char* section_name;  // Pointer to device constant string
        int valid;  // 1 if event is valid, 0 otherwise
    };

    // Per-warp event storage
    Event events[MAX_WARPS][MAX_EVENTS];
    int event_counts[MAX_WARPS];  // Number of events recorded per warp

    // Device pointer to per-block data
    WarpProfiler *block_data;
    
    // Host-side storage for managing memory
    int num_blocks;

    /**
     * @brief Start recording an event section. Call from warp leader only.
     * @param section_name Pointer to device constant string 
     */
    __device__ inline void start_event(const char* section_name) {
        unsigned int block_id;
        asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(block_id));
        
        WarpProfiler &block_profiler = block_data[block_id];
        
        unsigned int warp_id;
        asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp_id));
        
        if (warp_id >= MAX_WARPS) return;
        
        int idx = block_profiler.event_counts[warp_id];
        if (idx >= MAX_EVENTS) return;  // Overflow protection
        
        uint64_t timestamp;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
        
        block_profiler.events[warp_id][idx].start_time = timestamp;
        block_profiler.events[warp_id][idx].section_name = section_name;
        block_profiler.events[warp_id][idx].end_time = 0;
        block_profiler.events[warp_id][idx].valid = 0;  // Not valid until ended
    }

    /**
     * @brief End recording an event section. Call from warp leader only.
     * @param section_name Pointer to device constant string (should match start)
     */
    __device__ inline void end_event(const char* section_name) {
        unsigned int block_id;
        asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(block_id));
        
        WarpProfiler &block_profiler = block_data[block_id];
        
        unsigned int warp_id;
        asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp_id));
        
        if (warp_id >= MAX_WARPS) return;
        
        int idx = block_profiler.event_counts[warp_id];
        if (idx >= MAX_EVENTS) return;
        
        uint64_t timestamp;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
        
        // Verify this matches the current event being recorded
        if (block_profiler.events[warp_id][idx].section_name == section_name) {
            block_profiler.events[warp_id][idx].end_time = timestamp;
            block_profiler.events[warp_id][idx].valid = 1;
            block_profiler.event_counts[warp_id]++;
        }
    }

    /**
     * @brief Get the clock rate of the current GPU in kHz.
     * @return Clock rate in kHz
     */
    static inline double get_gpu_clock_rate_khz() {
        int device;
        cudaGetDevice(&device);
        
        int clock_rate_khz;
        cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device);
        
        return static_cast<double>(clock_rate_khz);
    }

    /**
     * @brief Export profiler data to Chrome Trace / Perfetto JSON format.
     * @param h_data Host profiler data array
     * @param num_blocks Number of blocks
     * @param filename Output JSON filename
     * @param clock_rate_khz GPU clock rate in kHz
     */
    static inline void export_chrome_trace(
        const WarpProfiler *h_data, 
        int num_blocks, 
        const std::string &filename,
        double clock_rate_khz
    ) {
        std::ofstream out(filename);
        if (!out.is_open()) {
            return;
        }

        // Collect all unique section name pointers and retrieve their strings
        std::set<const char*> unique_name_ptrs;
        for (int block = 0; block < num_blocks; block++) {
            const WarpProfiler &profiler = h_data[block];
            for (int warp = 0; warp < MAX_WARPS; warp++) {
                for (int evt = 0; evt < profiler.event_counts[warp]; evt++) {
                    const Event &event = profiler.events[warp][evt];
                    if (event.valid) {
                        unique_name_ptrs.insert(event.section_name);
                    }
                }
            }
        }

        // Retrieve the actual strings from device constant memory
        // Read byte-by-byte since we don't know the string length
        std::unordered_map<const char*, std::string> name_map;
        for (const char* dev_ptr : unique_name_ptrs) {
            char buffer[256];
            int len = 0;
            for (int i = 0; i < 255; i++) {
                cudaError_t err = cudaMemcpy(&buffer[i], dev_ptr + i, 1, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess || buffer[i] == '\0') {
                    break;
                }
                len++;
            }
            buffer[len] = '\0';
            
            if (len > 0) {
                name_map[dev_ptr] = std::string(buffer);
            } else {
                // Fallback to pointer address if reading fails
                char addr_str[32];
                snprintf(addr_str, sizeof(addr_str), "0x%llx", (unsigned long long)dev_ptr);
                name_map[dev_ptr] = std::string(addr_str);
            }
        }

        // Find the global minimum start time across all events to use as base
        uint64_t global_min_time = UINT64_MAX;
        for (int block = 0; block < num_blocks; block++) {
            const WarpProfiler &profiler = h_data[block];
            for (int warp = 0; warp < MAX_WARPS; warp++) {
                for (int evt = 0; evt < profiler.event_counts[warp]; evt++) {
                    const Event &event = profiler.events[warp][evt];
                    if (event.valid && event.start_time < global_min_time) {
                        global_min_time = event.start_time;
                    }
                }
            }
        }

        out << "[\n";
        bool first_event = true;

        for (int block = 0; block < num_blocks; block++) {
            const WarpProfiler &profiler = h_data[block];

            for (int warp = 0; warp < MAX_WARPS; warp++) {
                for (int evt = 0; evt < profiler.event_counts[warp]; evt++) {
                    const Event &event = profiler.events[warp][evt];
                    if (!event.valid) continue;

                    // Convert GPU timer ticks to microseconds relative to global start
                    double start_us = static_cast<double>(event.start_time - global_min_time) / clock_rate_khz * 1000.0;
                    double duration_us = static_cast<double>(event.end_time - event.start_time) / clock_rate_khz * 1000.0;

                    if (!first_event) {
                        out << ",\n";
                    }
                    first_event = false;

                    // Get the section name
                    std::string section_name = name_map[event.section_name];

                    // Chrome Trace Event Format
                    out << "  {\n";
                    out << "    \"name\": \"" << section_name << "\",\n";
                    out << "    \"cat\": \"kernel\",\n";
                    out << "    \"ph\": \"X\",\n";  // Complete event (duration)
                    out << "    \"ts\": " << start_us << ",\n";
                    out << "    \"dur\": " << duration_us << ",\n";
                    out << "    \"pid\": " << block << ",\n";  // Process = block
                    out << "    \"tid\": " << warp << "\n";   // Thread = warp
                    out << "  }";
                }
            }
        }

        out << "\n]\n";
        out.close();
    }
};

// ===== FREE FUNCTIONS FOR PROFILER API =====

/**
 * @brief Initialize a profiler instance. Allocates device memory for per-block data.
 * @param profiler Pointer to __device__ profiler instance
 * @param num_blocks Number of blocks that will be launched
 */
template <int MAX_EVENTS, int MAX_WARPS>
inline void profiler_init(WarpProfiler<MAX_EVENTS, MAX_WARPS> *profiler, int num_blocks) {
    WarpProfiler<MAX_EVENTS, MAX_WARPS> h_profiler;
    h_profiler.num_blocks = num_blocks;
    
    // Allocate device memory for per-block data
    cudaMalloc(&h_profiler.block_data, num_blocks * sizeof(WarpProfiler<MAX_EVENTS, MAX_WARPS>));
    cudaMemset(h_profiler.block_data, 0, num_blocks * sizeof(WarpProfiler<MAX_EVENTS, MAX_WARPS>));
    
    // Copy the profiler struct to the __device__ global
    cudaMemcpyToSymbol(*profiler, &h_profiler, sizeof(WarpProfiler<MAX_EVENTS, MAX_WARPS>));
}

/**
 * @brief Export profiler data to Chrome Trace format and cleanup all memory.
 * @param profiler Pointer to __device__ profiler instance
 * @param filename Output JSON filename (empty string to skip export)
 * @param clock_rate_khz GPU clock rate in kHz (0 to auto-detect)
 */
template <int MAX_EVENTS, int MAX_WARPS>
inline void profiler_export_and_cleanup(
    WarpProfiler<MAX_EVENTS, MAX_WARPS> *profiler,
    const std::string &filename = "",
    double clock_rate_khz = 0.0
) {
    // Copy profiler struct from device to host
    WarpProfiler<MAX_EVENTS, MAX_WARPS> h_profiler;
    cudaMemcpyFromSymbol(&h_profiler, *profiler, sizeof(WarpProfiler<MAX_EVENTS, MAX_WARPS>));
    
    if (!h_profiler.block_data) return;
    
    // Copy per-block data from device to host
    WarpProfiler<MAX_EVENTS, MAX_WARPS> *h_block_data = 
        new WarpProfiler<MAX_EVENTS, MAX_WARPS>[h_profiler.num_blocks];
    cudaMemcpy(h_block_data, h_profiler.block_data, 
               h_profiler.num_blocks * sizeof(WarpProfiler<MAX_EVENTS, MAX_WARPS>), 
               cudaMemcpyDeviceToHost);
    
    // Export if filename provided
    if (!filename.empty()) {
        if (clock_rate_khz == 0.0) {
            clock_rate_khz = WarpProfiler<MAX_EVENTS, MAX_WARPS>::get_gpu_clock_rate_khz();
        }
        WarpProfiler<MAX_EVENTS, MAX_WARPS>::export_chrome_trace(
            h_block_data, h_profiler.num_blocks, filename, clock_rate_khz);
    }
    
    // Cleanup
    delete[] h_block_data;
    cudaFree(h_profiler.block_data);
}
