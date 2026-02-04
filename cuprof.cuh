/*
 * cuprof - Single-file header-only CUDA kernel profiler
 *
 * Usage:
 *   1. Include this header in your CUDA code
 *   2. Call cuprof::init() before launching kernels
 *   3. Use CUPROF_START("name") and CUPROF_END("name") inside kernels
 *   4. Call cuprof::export_chrome_trace("output.json") after kernel completion
 *   5. Call cuprof::cleanup() when done
 *
 * Basic example:
 *   __global__ void my_kernel() {
 *       CUPROF_START("computation");
 *       // ... do work ...
 *       CUPROF_END("computation");
 *   }
 *
 * Warp-specialized kernels (only warp leader records):
 *   __global__ void warp_specialized_kernel() {
 *       int warp_id = threadIdx.x / 32;
 *       if (warp_id == 0) {
 *           CUPROF_WARP_START("producer");
 *           // ... producer work ...
 *           CUPROF_WARP_END("producer");
 *       } else {
 *           CUPROF_WARP_START("consumer");
 *           // ... consumer work ...
 *           CUPROF_WARP_END("consumer");
 *       }
 *   }
 *
 * Available macros:
 *   CUPROF_START(name)          - Record start (all threads)
 *   CUPROF_END(name)            - Record end (all threads)
 *   CUPROF_SCOPE(name)          - RAII scoped region (all threads)
 *   CUPROF_WARP_START(name)     - Record start (warp leader only)
 *   CUPROF_WARP_END(name)       - Record end (warp leader only)
 *   CUPROF_WARP_SCOPE(name)     - RAII scoped region (warp leader only)
 *   CUPROF_WARP_START_ID(name, warp_id) - Record with explicit warp ID
 *   CUPROF_WARP_END_ID(name, warp_id)   - Record with explicit warp ID
 *
 * MIT License - Copyright (c) 2026
 */

#ifndef CUPROF_CUH
#define CUPROF_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>

namespace cuprof {

// Configuration - override before including header if needed
#ifndef CUPROF_MAX_EVENTS
#define CUPROF_MAX_EVENTS (1 << 20)  // 1M events by default
#endif

// Event types
enum class EventType : uint8_t {
    START = 0,
    END = 1
};

// Profiling event structure (32 bytes, well-aligned)
struct Event {
    uint64_t timestamp;      // Nanoseconds from globaltimer
    uint32_t section_id;     // Hashed section name
    uint32_t block_id;       // Linear block index
    uint32_t thread_id;      // Linear thread index within block
    uint32_t sm_id;          // Streaming multiprocessor ID
    EventType type;          // START or END
    uint8_t warp_id;         // Warp ID within block
    uint8_t lane_id;         // Lane within warp
    uint8_t padding;         // Alignment padding
};

// Compile-time FNV-1a hash for section names
__host__ __device__ constexpr uint32_t hash_fnv1a(const char* str, uint32_t hash = 2166136261u) {
    return (*str == '\0') ? hash : hash_fnv1a(str + 1, (hash ^ static_cast<uint32_t>(*str)) * 16777619u);
}

// Global device state
namespace detail {
    // Device-side pointers and counters
    __device__ Event* d_events = nullptr;
    __device__ uint32_t d_event_count = 0;
    __device__ uint32_t d_max_events = 0;
    __device__ uint64_t d_base_time = 0;

    // Host-side state
    inline Event* h_events = nullptr;
    inline Event* d_events_ptr = nullptr;
    inline uint32_t* d_event_count_ptr = nullptr;
    inline uint32_t h_max_events = 0;
    inline bool initialized = false;

    // Section name registry (hash -> name)
    inline std::unordered_map<uint32_t, std::string> section_names;
}

// Device function to read globaltimer (nanoseconds since device boot)
// This provides consistent timing across all SMs, unlike clock64() which is per-SM
__device__ __forceinline__ uint64_t get_globaltimer() {
    uint64_t time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

// Device function to get SM ID
__device__ __forceinline__ uint32_t get_sm_id() {
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    return sm_id;
}

// Device function to get warp ID within block
__device__ __forceinline__ uint32_t get_warp_id() {
    uint32_t warp_id;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp_id));
    return warp_id;
}

// Device function to get lane ID within warp
__device__ __forceinline__ uint32_t get_lane_id() {
    uint32_t lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

// Device function to record an event
__device__ __forceinline__ void record_event(uint32_t section_id, EventType type) {
    if (detail::d_events == nullptr) return;

    // Atomically allocate an event slot
    uint32_t idx = atomicAdd(&detail::d_event_count, 1);
    if (idx >= detail::d_max_events) {
        // Buffer full, undo the increment and return
        atomicSub(&detail::d_event_count, 1);
        return;
    }

    // Fill in event data
    Event& evt = detail::d_events[idx];
    evt.timestamp = get_globaltimer() - detail::d_base_time;
    evt.section_id = section_id;
    evt.block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    evt.thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    evt.sm_id = get_sm_id();
    evt.type = type;
    evt.warp_id = static_cast<uint8_t>(get_warp_id());
    evt.lane_id = static_cast<uint8_t>(get_lane_id());
}

// Device function to check if current thread is warp leader (lane 0)
__device__ __forceinline__ bool is_warp_leader() {
    return get_lane_id() == 0;
}

// Device function to record an event (warp-leader only variant)
// For warp-specialized kernels, use warp_id parameter to identify which warp
__device__ __forceinline__ void record_event_warp(uint32_t section_id, EventType type, uint32_t warp_id) {
    if (detail::d_events == nullptr) return;
    if (!is_warp_leader()) return;  // Only warp leader records

    // Atomically allocate an event slot
    uint32_t idx = atomicAdd(&detail::d_event_count, 1);
    if (idx >= detail::d_max_events) {
        atomicSub(&detail::d_event_count, 1);
        return;
    }

    // Fill in event data
    Event& evt = detail::d_events[idx];
    evt.timestamp = get_globaltimer() - detail::d_base_time;
    evt.section_id = section_id;
    evt.block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    evt.thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    evt.sm_id = get_sm_id();
    evt.type = type;
    evt.warp_id = static_cast<uint8_t>(warp_id);
    evt.lane_id = 0;  // Always 0 since only warp leader records
}

// Device macros for recording with compile-time string hashing
#define CUPROF_START(name) do { \
    constexpr uint32_t _cuprof_hash = ::cuprof::hash_fnv1a(name); \
    ::cuprof::record_event(_cuprof_hash, ::cuprof::EventType::START); \
} while(0)

#define CUPROF_END(name) do { \
    constexpr uint32_t _cuprof_hash = ::cuprof::hash_fnv1a(name); \
    ::cuprof::record_event(_cuprof_hash, ::cuprof::EventType::END); \
} while(0)

// Warp-leader-only macros for warp-specialized kernels
// These only record from lane 0 of each warp, reducing overhead and event count
#define CUPROF_WARP_START(name) do { \
    constexpr uint32_t _cuprof_hash = ::cuprof::hash_fnv1a(name); \
    if (::cuprof::is_warp_leader()) { \
        ::cuprof::record_event(_cuprof_hash, ::cuprof::EventType::START); \
    } \
} while(0)

#define CUPROF_WARP_END(name) do { \
    constexpr uint32_t _cuprof_hash = ::cuprof::hash_fnv1a(name); \
    if (::cuprof::is_warp_leader()) { \
        ::cuprof::record_event(_cuprof_hash, ::cuprof::EventType::END); \
    } \
} while(0)

// Warp-specialized macros with explicit warp ID for warp-specialized kernels
// Use when different warps perform different roles (e.g., producer/consumer)
#define CUPROF_WARP_START_ID(name, warp_id) do { \
    constexpr uint32_t _cuprof_hash = ::cuprof::hash_fnv1a(name); \
    ::cuprof::record_event_warp(_cuprof_hash, ::cuprof::EventType::START, warp_id); \
} while(0)

#define CUPROF_WARP_END_ID(name, warp_id) do { \
    constexpr uint32_t _cuprof_hash = ::cuprof::hash_fnv1a(name); \
    ::cuprof::record_event_warp(_cuprof_hash, ::cuprof::EventType::END, warp_id); \
} while(0)

// Kernel to capture base time (run with 1 thread before other kernels)
__global__ void init_base_time_kernel(uint64_t* out_base_time) {
    *out_base_time = get_globaltimer();
}

// Host function to register a section name (call before export for readable names)
inline void register_section(const char* name) {
    uint32_t hash = hash_fnv1a(name);
    detail::section_names[hash] = name;
}

// Host function to initialize profiler
inline bool init(uint32_t max_events = CUPROF_MAX_EVENTS) {
    if (detail::initialized) {
        fprintf(stderr, "cuprof: already initialized\n");
        return false;
    }

    // Allocate event buffer on device
    cudaError_t err = cudaMalloc(&detail::d_events_ptr, max_events * sizeof(Event));
    if (err != cudaSuccess) {
        fprintf(stderr, "cuprof: failed to allocate event buffer: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaMemset(detail::d_events_ptr, 0, max_events * sizeof(Event));

    // Allocate event counter on device
    err = cudaMalloc(&detail::d_event_count_ptr, sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(detail::d_events_ptr);
        fprintf(stderr, "cuprof: failed to allocate event counter: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaMemset(detail::d_event_count_ptr, 0, sizeof(uint32_t));

    // Allocate temporary storage for base time
    uint64_t* d_base_time_ptr;
    err = cudaMalloc(&d_base_time_ptr, sizeof(uint64_t));
    if (err != cudaSuccess) {
        cudaFree(detail::d_events_ptr);
        cudaFree(detail::d_event_count_ptr);
        fprintf(stderr, "cuprof: failed to allocate base time: %s\n", cudaGetErrorString(err));
        return false;
    }

    // Initialize base time on device
    init_base_time_kernel<<<1, 1>>>(d_base_time_ptr);
    cudaDeviceSynchronize();

    // Copy base time to device symbol
    uint64_t h_base_time;
    cudaMemcpy(&h_base_time, d_base_time_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(detail::d_base_time, &h_base_time, sizeof(uint64_t));
    cudaFree(d_base_time_ptr);

    // Initialize device symbols
    cudaMemcpyToSymbol(detail::d_events, &detail::d_events_ptr, sizeof(Event*));
    uint32_t zero = 0;
    cudaMemcpyToSymbol(detail::d_event_count, &zero, sizeof(uint32_t));
    cudaMemcpyToSymbol(detail::d_max_events, &max_events, sizeof(uint32_t));

    // Allocate host buffer for events
    detail::h_events = new Event[max_events];
    detail::h_max_events = max_events;
    detail::initialized = true;

    return true;
}

// Host function to copy events from device to host
inline uint32_t fetch_events() {
    if (!detail::initialized) return 0;

    // Get current event count
    uint32_t count = 0;
    cudaMemcpyFromSymbol(&count, detail::d_event_count, sizeof(uint32_t));

    if (count > detail::h_max_events) {
        count = detail::h_max_events;
    }

    if (count > 0) {
        cudaMemcpy(detail::h_events, detail::d_events_ptr, count * sizeof(Event), cudaMemcpyDeviceToHost);
    }

    return count;
}

// Host function to reset profiler state (clear events, keep initialized)
inline void reset() {
    if (!detail::initialized) return;

    // Reset event counter
    uint32_t zero = 0;
    cudaMemcpyToSymbol(detail::d_event_count, &zero, sizeof(uint32_t));

    // Reset base time
    uint64_t* d_base_time_ptr;
    cudaMalloc(&d_base_time_ptr, sizeof(uint64_t));
    init_base_time_kernel<<<1, 1>>>(d_base_time_ptr);
    cudaDeviceSynchronize();

    uint64_t h_base_time;
    cudaMemcpy(&h_base_time, d_base_time_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(detail::d_base_time, &h_base_time, sizeof(uint64_t));
    cudaFree(d_base_time_ptr);
}

// Host function to export to Chrome Trace / Perfetto format
inline bool export_chrome_trace(const char* filename) {
    if (!detail::initialized) {
        fprintf(stderr, "cuprof: not initialized\n");
        return false;
    }

    // Ensure all kernels have completed
    cudaDeviceSynchronize();
    uint32_t count = fetch_events();

    if (count == 0) {
        fprintf(stderr, "cuprof: no events to export\n");
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "cuprof: failed to open file: %s\n", filename);
        return false;
    }

    // Chrome trace format uses microseconds, globaltimer gives nanoseconds
    constexpr double ns_to_us = 0.001;

    file << "{\"traceEvents\":[\n";

    bool first = true;
    for (uint32_t i = 0; i < count; i++) {
        const Event& evt = detail::h_events[i];

        // Get section name from registry or use hash as fallback
        std::string name;
        auto it = detail::section_names.find(evt.section_id);
        if (it != detail::section_names.end()) {
            name = it->second;
        } else {
            char buf[32];
            snprintf(buf, sizeof(buf), "section_%08x", evt.section_id);
            name = buf;
        }

        // Escape JSON special characters
        std::string escaped_name;
        for (char c : name) {
            if (c == '"') escaped_name += "\\\"";
            else if (c == '\\') escaped_name += "\\\\";
            else if (c == '\n') escaped_name += "\\n";
            else if (c == '\r') escaped_name += "\\r";
            else if (c == '\t') escaped_name += "\\t";
            else escaped_name += c;
        }

        // Convert timestamp to microseconds
        double ts_us = evt.timestamp * ns_to_us;

        // Phase: B = begin, E = end
        char phase = (evt.type == EventType::START) ? 'B' : 'E';

        if (!first) file << ",\n";
        first = false;

        // Use SM ID as process ID, block_id * 1024 + thread_id as thread ID
        // This groups events by SM in the Perfetto UI
        file << "{\"name\":\"" << escaped_name << "\""
             << ",\"cat\":\"cuda\""
             << ",\"ph\":\"" << phase << "\""
             << ",\"ts\":" << std::fixed << ts_us
             << ",\"pid\":" << evt.sm_id
             << ",\"tid\":" << (evt.block_id * 1024 + evt.thread_id)
             << ",\"args\":{\"block\":" << evt.block_id
             << ",\"thread\":" << evt.thread_id
             << ",\"warp\":" << static_cast<int>(evt.warp_id)
             << ",\"lane\":" << static_cast<int>(evt.lane_id) << "}}";
    }

    file << "\n],\n";

    // Add metadata
    file << "\"displayTimeUnit\":\"ns\"\n";
    file << "}\n";

    file.close();
    printf("cuprof: exported %u events to %s\n", count, filename);
    return true;
}

// Statistics structure
struct Stats {
    uint32_t total_events;
    uint32_t unique_sections;
    uint64_t min_timestamp_ns;
    uint64_t max_timestamp_ns;
    double duration_us;
};

// Host function to get profiling statistics
inline Stats get_stats() {
    Stats stats = {0, 0, UINT64_MAX, 0, 0.0};

    if (!detail::initialized) return stats;

    cudaDeviceSynchronize();
    stats.total_events = fetch_events();

    std::unordered_map<uint32_t, bool> seen_sections;

    for (uint32_t i = 0; i < stats.total_events; i++) {
        const Event& evt = detail::h_events[i];
        seen_sections[evt.section_id] = true;
        stats.min_timestamp_ns = std::min(stats.min_timestamp_ns, evt.timestamp);
        stats.max_timestamp_ns = std::max(stats.max_timestamp_ns, evt.timestamp);
    }

    stats.unique_sections = seen_sections.size();

    if (stats.total_events == 0) {
        stats.min_timestamp_ns = 0;
    } else {
        stats.duration_us = (stats.max_timestamp_ns - stats.min_timestamp_ns) / 1000.0;
    }

    return stats;
}

// Host function to cleanup and free resources
inline void cleanup() {
    if (!detail::initialized) return;

    cudaFree(detail::d_events_ptr);
    cudaFree(detail::d_event_count_ptr);
    delete[] detail::h_events;

    detail::d_events_ptr = nullptr;
    detail::d_event_count_ptr = nullptr;
    detail::h_events = nullptr;
    detail::h_max_events = 0;
    detail::initialized = false;
    detail::section_names.clear();
}

// RAII helper for automatic initialization and cleanup
class ScopedProfiler {
public:
    explicit ScopedProfiler(uint32_t max_events = CUPROF_MAX_EVENTS) {
        init(max_events);
    }
    ~ScopedProfiler() {
        cleanup();
    }
    ScopedProfiler(const ScopedProfiler&) = delete;
    ScopedProfiler& operator=(const ScopedProfiler&) = delete;
};

// Device-side RAII helper for scoped profiling regions
template<uint32_t SECTION_HASH>
class ScopedRegion {
public:
    __device__ ScopedRegion() {
        record_event(SECTION_HASH, EventType::START);
    }
    __device__ ~ScopedRegion() {
        record_event(SECTION_HASH, EventType::END);
    }
};

// Macro for scoped region profiling (automatically records start/end)
#define CUPROF_SCOPE(name) \
    ::cuprof::ScopedRegion<::cuprof::hash_fnv1a(name)> _cuprof_scope_##__LINE__

// Device-side RAII helper for warp-leader-only scoped profiling
template<uint32_t SECTION_HASH>
class ScopedWarpRegion {
public:
    __device__ ScopedWarpRegion() {
        if (is_warp_leader()) {
            record_event(SECTION_HASH, EventType::START);
        }
    }
    __device__ ~ScopedWarpRegion() {
        if (is_warp_leader()) {
            record_event(SECTION_HASH, EventType::END);
        }
    }
};

// Macro for warp-leader-only scoped region profiling
#define CUPROF_WARP_SCOPE(name) \
    ::cuprof::ScopedWarpRegion<::cuprof::hash_fnv1a(name)> _cuprof_warp_scope_##__LINE__

} // namespace cuprof

#endif // CUPROF_CUH
