#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <limits>
#include <iomanip>

namespace cuprof {

// Forward declaration
template <int MAX_EVENTS, int MAX_WARPS> struct Profiler;

// Block-wide state containing all per-block profiling data (in global memory)
template <int MAX_EVENTS = 128, int MAX_WARPS = 32>
struct BlockState {
    struct EventData {
        uint64_t start_time_ns;      // globaltimer nanoseconds (for cross-SM sync)
        uint64_t start_time_clock;   // clock64 cycles (for precise duration)
        uint64_t end_time_clock;     // clock64 cycles (for precise duration)
        const char* section_name;    // Pointer to device constant string
        unsigned int smid;  // SM ID where this event was recorded
    };

    // Per-warp event storage
    EventData events[MAX_WARPS][MAX_EVENTS];
    int event_counts[MAX_WARPS];  // Number of events recorded per warp
    
    // Block initialization - we don't actually need this anymore since we store absolute times
    int initialized;
};

// Helper to check if current thread is warp leader
__device__ __forceinline__ bool is_warp_leader() {
    unsigned int lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id == 0;
}


// Event handle stores all data locally - no gmem writes until end()
struct Event {
    const char* section_name;
    uint64_t start_time_ns;
    uint64_t start_time_clock;
    unsigned int smid;
    
    __device__ __host__ Event() : section_name(nullptr), 
                                   start_time_ns(0), start_time_clock(0), smid(0) {}
    __device__ __host__ bool is_valid() const { return section_name != nullptr; }
};

/**
 * @brief Lightweight per-warp profiler for CUDA kernels with Chrome Trace/Perfetto export.
 *
 * Usage:
 *
 * 1. Define a __device__ global profiler instance:
 *
 *      __device__ cuprof::Profiler<> myprofiler;
 *
 * 2. Host side - initialize:
 *
 *      cuprof::init(&myprofiler, num_blocks);
 *
 * 3. Device side - initialize and record events:
 *
 *      myprofiler.block_init();  // Call once at kernel start
 *
 *      if (cuprof::is_warp_leader()) {
 *          cuprof::Event e = myprofiler.start("compute");
 *          // ... work ...
 *          myprofiler.end(e);
 *      }
 *
 * 4. Host side - export and cleanup:
 *
 *      cuprof::export_and_cleanup(&myprofiler, "trace.json");
 */
template <int MAX_EVENTS = 128, int MAX_WARPS = 32>
struct __align__(16) Profiler {
    // Device pointer to per-block state array
    BlockState<MAX_EVENTS, MAX_WARPS> *block_states;
    
    // Host-side storage for managing memory
    int num_blocks;

    /**
     * @brief Initialize block state. Call once per block, before any profiling.
     * Should be called by one thread (e.g., threadIdx.x == 0) with a __syncthreads() after.
     */
    __device__ inline void block_init() {
        unsigned int block_id;
        asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(block_id));
        
        BlockState<MAX_EVENTS, MAX_WARPS> &state = block_states[block_id];
        
        if (threadIdx.x == 0) {
            state.initialized = 1;
        }
        __syncthreads();
    }

    /**
     * @brief Start recording an event section. Call from warp leader only.
     * No gmem writes - all data stored in returned Event handle.
     * @param section_name Pointer to device constant string
     * @return Event handle to pass to end() 
     */
    __device__ inline Event start(const char* section_name) {
        // Capture both globaltimer (for cross-SM sync) and clock64 (for precise duration)
        uint64_t global_time, clock_time;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(global_time));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock_time));
        
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        
        // Return event with all data in registers
        Event e;
        e.section_name = section_name;
        e.start_time_ns = global_time;
        e.start_time_clock = clock_time;
        e.smid = smid;
        return e;
    }

    /**
     * @brief End recording an event section. Call from warp leader only.
     * Writes all event data to gmem in one operation.
     * @param event Event handle returned by start()
     */
    __device__ inline void end(Event event) {
        if (!event.is_valid()) return;  // Invalid event
        
        unsigned int block_id;
        asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(block_id));
        
        BlockState<MAX_EVENTS, MAX_WARPS> &state = block_states[block_id];
        
        unsigned int warp_id;
        asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp_id));
        
        if (warp_id >= MAX_WARPS) return;
        
        // Atomically reserve an index for this event
        int idx = atomicAdd(&state.event_counts[warp_id], 1);
        if (idx >= MAX_EVENTS) return;  // Overflow protection
        
        uint64_t end_clock;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock));
        
        // Write all event data to gmem in one go
        typename BlockState<MAX_EVENTS, MAX_WARPS>::EventData &evt = state.events[warp_id][idx];
        evt.start_time_ns = event.start_time_ns;
        evt.start_time_clock = event.start_time_clock;
        evt.end_time_clock = end_clock;
        evt.section_name = event.section_name;
        evt.smid = event.smid;
    }

    /**
     * @brief Export profiler data to Chrome Trace / Perfetto JSON format.
     * @param h_block_states Host block state array
     * @param num_blocks Number of blocks
     * @param filename Output JSON filename
     */
    static inline void export_chrome_trace(
        const BlockState<MAX_EVENTS, MAX_WARPS> *h_block_states, 
        int num_blocks, 
        const std::string &filename
    ) {
        std::ofstream out(filename);
        if (!out.is_open()) {
            return;
        }

        // Collect all unique section name pointers and retrieve their strings
        std::set<const char*> unique_name_ptrs;
        for (int block = 0; block < num_blocks; block++) {
            const BlockState<MAX_EVENTS, MAX_WARPS> &state = h_block_states[block];
            for (int warp = 0; warp < MAX_WARPS; warp++) {
                for (int evt = 0; evt < state.event_counts[warp]; evt++) {
                    const typename BlockState<MAX_EVENTS, MAX_WARPS>::EventData &event = state.events[warp][evt];
                    if (event.section_name != nullptr) {
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

        // Get GPU clock rate for cycle-to-time conversion
        int device;
        cudaGetDevice(&device);
        int clock_rate_khz;
        cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device);
        
        // Find minimum start_time_ns across all events (globaltimer is synchronized across SMs)
        uint64_t global_min_time_ns = UINT64_MAX;
        for (int block = 0; block < num_blocks; block++) {
            const BlockState<MAX_EVENTS, MAX_WARPS> &state = h_block_states[block];
            for (int warp = 0; warp < MAX_WARPS; warp++) {
                for (int evt = 0; evt < state.event_counts[warp]; evt++) {
                    const typename BlockState<MAX_EVENTS, MAX_WARPS>::EventData &event = state.events[warp][evt];
                    if (event.section_name != nullptr && event.start_time_ns > 0) {
                        global_min_time_ns = std::min(global_min_time_ns, event.start_time_ns);
                    }
                }
            }
        }
        
        // If no valid events found, set to 0
        if (global_min_time_ns == UINT64_MAX) {
            global_min_time_ns = 0;
        }

        out << "[\n";
        bool first_event = true;

        for (int block = 0; block < num_blocks; block++) {
            const BlockState<MAX_EVENTS, MAX_WARPS> &state = h_block_states[block];

            for (int warp = 0; warp < MAX_WARPS; warp++) {
                for (int evt = 0; evt < state.event_counts[warp]; evt++) {
                    const typename BlockState<MAX_EVENTS, MAX_WARPS>::EventData &event = state.events[warp][evt];
                    if (event.section_name == nullptr) continue;

                    // Use globaltimer for timestamp (cross-SM synchronized), clock64 for duration
                    // Timestamp: globaltimer normalized to minimum (nanoseconds -> microseconds)
                    double start_us = static_cast<double>(event.start_time_ns - global_min_time_ns) / 1000.0;
                    
                    // Use clock64 for precise duration measurement within same SM
                    // clock_rate_khz is in kHz, so cycles * 1000.0 / clock_rate_khz = microseconds
                    double duration_us;
                    if (event.end_time_clock >= event.start_time_clock) {
                        uint64_t duration_cycles = event.end_time_clock - event.start_time_clock;
                        duration_us = static_cast<double>(duration_cycles) * 1000.0 / clock_rate_khz;
                        
                        // Sanity check: if duration is unreasonably large (>1 second), likely corrupted data
                        if (duration_us > 1000000.0) {  // > 1 second
                            duration_us = 0.0;
                        }
                    } else {
                        // clock64 wraparound - should be extremely rare, treat as corrupted
                        duration_us = 0.0;
                    }

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
                    // Compute global warp ID: block_id * warps_per_block + local_warp_id
                    int global_warp_id = block * MAX_WARPS + warp;
                    
                    out << "    \"ts\": " << std::fixed << std::setprecision(6) << start_us << ",\n";
                    out << "    \"dur\": " << std::fixed << std::setprecision(6) << duration_us << ",\n";
                    out << "    \"pid\": " << event.smid << ",\n";  // Process = SM ID
                    out << "    \"tid\": " << global_warp_id << ",\n";   // Thread = global warp ID
                    out << "    \"args\": {\"block\": " << block << ", \"warp\": " << warp << ", \"smid\": " << event.smid << "}\n";
                    out << "  }";
                }
            }
        }

        out << "\n]\n";
        out.close();
    }
};

/**
 * @brief Initialize a profiler instance. Allocates device memory for per-block state.
 * @param profiler Pointer to __device__ profiler instance
 * @param num_blocks Number of blocks that will be launched
 */
template <int MAX_EVENTS, int MAX_WARPS>
inline void init(Profiler<MAX_EVENTS, MAX_WARPS> *profiler, int num_blocks) {
    Profiler<MAX_EVENTS, MAX_WARPS> h_profiler;
    h_profiler.num_blocks = num_blocks;
    
    // Allocate device memory for per-block state array
    cudaMalloc(&h_profiler.block_states, num_blocks * sizeof(BlockState<MAX_EVENTS, MAX_WARPS>));
    cudaMemset(h_profiler.block_states, 0, num_blocks * sizeof(BlockState<MAX_EVENTS, MAX_WARPS>));
    
    // Copy the profiler struct to the __device__ global
    cudaMemcpyToSymbol(*profiler, &h_profiler, sizeof(Profiler<MAX_EVENTS, MAX_WARPS>));
}

/**
 * @brief Export profiler data to Chrome Trace format and cleanup all memory.
 * @param profiler Pointer to __device__ profiler instance
 * @param filename Output JSON filename (empty string to skip export)
 */
template <int MAX_EVENTS, int MAX_WARPS>
inline void export_and_cleanup(
    Profiler<MAX_EVENTS, MAX_WARPS> *profiler,
    const std::string &filename = ""
) {
    // Copy profiler struct from device to host
    Profiler<MAX_EVENTS, MAX_WARPS> h_profiler;
    cudaMemcpyFromSymbol(&h_profiler, *profiler, sizeof(Profiler<MAX_EVENTS, MAX_WARPS>));
    
    if (!h_profiler.block_states) return;
    
    // Copy per-block state from device to host
    BlockState<MAX_EVENTS, MAX_WARPS> *h_block_states = 
        new BlockState<MAX_EVENTS, MAX_WARPS>[h_profiler.num_blocks];
    cudaMemcpy(h_block_states, h_profiler.block_states, 
               h_profiler.num_blocks * sizeof(BlockState<MAX_EVENTS, MAX_WARPS>), 
               cudaMemcpyDeviceToHost);
    
    // Export if filename provided
    if (!filename.empty()) {
        Profiler<MAX_EVENTS, MAX_WARPS>::export_chrome_trace(
            h_block_states, h_profiler.num_blocks, filename);
    }
    
    // Cleanup
    delete[] h_block_states;
    cudaFree(h_profiler.block_states);
}

} // namespace cuprof
