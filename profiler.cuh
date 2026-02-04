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
    unsigned int lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id == 0;
}


// Event ID type for type safety and clarity
struct EventId {
    int id;
    
    __device__ __host__ EventId() : id(-1) {}
    __device__ __host__ explicit EventId(int i) : id(i) {}
    __device__ __host__ bool is_valid() const { return id >= 0; }
};

/**
 * @brief Lightweight per-warp profiler for CUDA kernels with Chrome Trace/Perfetto export.
 *
 * Usage:
 *
 * 1. Define a __device__ global profiler instance:
 *
 *      __device__ WarpProfiler<> myprofiler;
 *
 * 2. Host side - initialize:
 *
 *      profiler_init(&myprofiler, num_blocks);
 *
 * 3. Device side - record events from warp leader only:
 *
 *      if (profiler_is_warp_leader()) {
 *          EventId id = myprofiler.start_event("compute");
 *          // ... work ...
 *          myprofiler.end_event(id);
 *      }
 *
 * 4. Host side - export and cleanup:
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
        unsigned int smid;  // SM ID where this event was recorded
        unsigned int block_id;  // Block ID for reference
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
     * @return Event ID to pass to end_event() 
     */
    __device__ inline EventId start_event(const char* section_name) {
        unsigned int block_id;
        asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(block_id));
        
        WarpProfiler &block_profiler = block_data[block_id];
        
        unsigned int warp_id;
        asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp_id));
        
        if (warp_id >= MAX_WARPS) return EventId();
        
        int idx = block_profiler.event_counts[warp_id];
        if (idx >= MAX_EVENTS) return EventId();  // Overflow protection
        
        uint64_t timestamp;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(timestamp));
        
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        
        block_profiler.events[warp_id][idx].start_time = timestamp;
        block_profiler.events[warp_id][idx].section_name = section_name;
        block_profiler.events[warp_id][idx].end_time = 0;
        block_profiler.events[warp_id][idx].valid = 0;  // Not valid until ended
        block_profiler.events[warp_id][idx].smid = smid;
        block_profiler.events[warp_id][idx].block_id = block_id;
        
        // Increment counter and return the ID for this event
        block_profiler.event_counts[warp_id]++;
        return EventId(idx);
    }

    /**
     * @brief End recording an event section. Call from warp leader only.
     * @param event_id Event ID returned by start_event()
     */
    __device__ inline void end_event(EventId event_id) {
        if (!event_id.is_valid()) return;  // Invalid event ID
        
        unsigned int block_id;
        asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(block_id));
        
        WarpProfiler &block_profiler = block_data[block_id];
        
        unsigned int warp_id;
        asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp_id));
        
        if (warp_id >= MAX_WARPS) return;
        if (event_id.id >= MAX_EVENTS) return;
        
        uint64_t timestamp;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(timestamp));
        
        // Directly access the event by ID - no search needed
        block_profiler.events[warp_id][event_id.id].end_time = timestamp;
        block_profiler.events[warp_id][event_id.id].valid = 1;
    }

    /**
     * @brief Export profiler data to Chrome Trace / Perfetto JSON format.
     * @param h_data Host profiler data array
     * @param num_blocks Number of blocks
     * @param filename Output JSON filename
     */
    static inline void export_chrome_trace(
        const WarpProfiler *h_data, 
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

        // Get GPU clock rate for cycle-to-time conversion
        int device;
        cudaGetDevice(&device);
        int clock_rate_khz;
        cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device);
        
        // Find per-SM minimum start time (since clock64 is per-SM)
        std::unordered_map<unsigned int, uint64_t> sm_min_time;
        for (int block = 0; block < num_blocks; block++) {
            const WarpProfiler &profiler = h_data[block];
            for (int warp = 0; warp < MAX_WARPS; warp++) {
                for (int evt = 0; evt < profiler.event_counts[warp]; evt++) {
                    const Event &event = profiler.events[warp][evt];
                    if (event.valid) {
                        if (sm_min_time.find(event.smid) == sm_min_time.end()) {
                            sm_min_time[event.smid] = event.start_time;
                        } else {
                            sm_min_time[event.smid] = std::min(sm_min_time[event.smid], event.start_time);
                        }
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

                    // Convert clock64 cycles to microseconds relative to per-SM start
                    // clock_rate_khz is in kHz, so cycles * 1000.0 / clock_rate_khz = microseconds
                    uint64_t sm_base = sm_min_time[event.smid];
                    double start_us = static_cast<double>(event.start_time - sm_base) * 1000.0 / clock_rate_khz;
                    
                    // Handle potential clock64 wraparound and sanity check
                    // If end_time < start_time, the counter wrapped around
                    double duration_us;
                    if (event.end_time >= event.start_time) {
                        uint64_t duration_cycles = event.end_time - event.start_time;
                        duration_us = static_cast<double>(duration_cycles) * 1000.0 / clock_rate_khz;
                        
                        // Sanity check: if duration is unreasonably large (>1 second), likely corrupted data
                        // This can happen if events aren't properly paired or clock64 has discontinuities
                        if (duration_us > 1000000.0) {  // > 1 second
                            duration_us = 0.0;
                        }
                    } else {
                        // Wraparound occurred - duration would be negative, so clamp to 0
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
                    out << "    \"ts\": " << start_us << ",\n";
                    out << "    \"dur\": " << duration_us << ",\n";
                    out << "    \"pid\": " << event.smid << ",\n";  // Process = SM
                    out << "    \"tid\": " << warp << ",\n";   // Thread = warp
                    out << "    \"args\": {\"block\": " << event.block_id << ", \"smid\": " << event.smid << "}\n";
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
 */
template <int MAX_EVENTS, int MAX_WARPS>
inline void profiler_export_and_cleanup(
    WarpProfiler<MAX_EVENTS, MAX_WARPS> *profiler,
    const std::string &filename = ""
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
        WarpProfiler<MAX_EVENTS, MAX_WARPS>::export_chrome_trace(
            h_block_data, h_profiler.num_blocks, filename);
    }
    
    // Cleanup
    delete[] h_block_data;
    cudaFree(h_profiler.block_data);
}
