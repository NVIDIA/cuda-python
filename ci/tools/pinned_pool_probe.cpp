// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Standalone driver-only reproducer. See pinned_pool_probe_native.md for build/run instructions.
#include <cuda.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
constexpr size_t allocation_bytes = 1 << 20;

// The parent kills a hung process after 120s. A shared file mapping retains the
// last API boundary without a flush/system call in the allocation loop.
class Progress {
    static constexpr size_t size = 1024;
    char *data = nullptr;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int file = -1;
#endif
   public:
    explicit Progress(const fs::path &path)
    {
#ifdef _WIN32
        file = CreateFileW(
            path.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ,
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (file != INVALID_HANDLE_VALUE) {
            mapping = CreateFileMappingW(file, nullptr, PAGE_READWRITE, 0, size, nullptr);
            if (mapping)
                data = static_cast<char *>(MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, size));
        }
#else
        file = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (file >= 0 && ftruncate(file, size) == 0) {
            void *view = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
            if (view != MAP_FAILED)
                data = static_cast<char *>(view);
        }
#endif
        if (!data)
            throw std::runtime_error("Cannot map progress.json");
        std::memset(data, ' ', size);
    }
    ~Progress()
    {
#ifdef _WIN32
        if (data)
            UnmapViewOfFile(data);
        if (mapping)
            CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE)
            CloseHandle(file);
#else
        if (data)
            munmap(data, size);
        if (file >= 0)
            close(file);
#endif
    }
    void
    update(int iteration, int completed, const char *stage, const char *api, int status, bool returned)
    {
        char buffer[size];
        int length = std::snprintf(
            buffer,
            size,
            "{\"iteration\":%d,\"completed_cycles\":%d,\"stage\":\"%s\",\"api\":\"%s\","
            "\"status\":%d,\"returned\":%s}",
            iteration,
            completed,
            stage,
            api,
            status,
            returned ? "true" : "false"
        );
        std::memcpy(data, buffer, length);
        std::memset(data + length, ' ', size - length);
    }
};

struct Event {
    int iteration = -1;
    const char *stage = "startup";
    const char *api = "none";
    int status = 0;
};

struct Probe {
    Progress progress;
    std::array<Event, 128> recent{};
    size_t calls = 0;
    int iteration = -1;
    int completed = 0;
    Event failure{};
    bool failed = false;
    cuuint64_t high_before = 0, current_before = 0, high_after = 0, current_after = 0;

    explicit Probe(const fs::path &output) : progress(output / "progress.json") {}

    template <typename F>
    void
    call(const char *stage, const char *api, F invoke)
    {
        progress.update(iteration, completed, stage, api, -1, false);
        CUresult status = invoke();
        Event event{iteration, stage, api, static_cast<int>(status)};
        recent[calls++ % recent.size()] = event;
        progress.update(iteration, completed, stage, api, event.status, true);
        if (status != CUDA_SUCCESS) {
            failure = event;
            failed = true;
            throw event;
        }
    }
    void
    dump(const fs::path &output) const
    {
        std::ofstream log(output / "calls.jsonl");
        log.exceptions(std::ios::badbit | std::ios::failbit);
        size_t start = calls > recent.size() ? calls - recent.size() : 0;
        for (size_t i = start; i < calls; ++i) {
            const auto &event = recent[i % recent.size()];
            log << "{\"iteration\":" << event.iteration << ",\"stage\":\"" << event.stage << "\",\"api\":\""
                << event.api << "\",\"status\":" << event.status << "}\n";
        }
    }
};

// Evaluate the CUDA call only after recording its entry boundary.
#define CALL(stage, api, ...) probe.call(stage, #api, [&] { return api(__VA_ARGS__); })

struct Options {
    fs::path output = "native-pinned-memory";
    std::string stream = "legacy";
    std::string variant = "baseline";
    int cycles = 200;
    bool revoke = false;
};

Options
parse(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--revoke-access")
            options.revoke = true;
        else {
            if (++i == argc)
                throw std::runtime_error("Missing option value");
            std::string value = argv[i];
            if (key == "--stream")
                options.stream = value;
            else if (key == "--variant")
                options.variant = value;
            else if (key == "--output-dir")
                options.output = value;
            else if (key == "--cycles") {
                size_t end = 0;
                options.cycles = std::stoi(value, &end);
                if (end != value.size())
                    throw std::runtime_error("Invalid cycle count");
            }
            else
                throw std::runtime_error("Unknown option: " + key);
        }
    }
    if (options.cycles < 1 || options.cycles > 10000)
        throw std::runtime_error("Cycles must be in [1, 10000]");
    if (options.stream != "legacy" && options.stream != "per-thread" && options.stream != "nonblocking")
        throw std::runtime_error("Stream must be legacy, per-thread, or nonblocking");
    if (options.variant != "baseline" && options.variant != "no-trim" && options.variant != "no-reset" &&
        options.variant != "reuse")
        throw std::runtime_error("Variant must be baseline, no-trim, no-reset, or reuse");
    return options;
}

void
memory_snapshot(const char *phase, bool context_current)
{
    std::cout << "memory phase=" << phase;
#ifdef _WIN32
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory)) {
        std::cout << " available_physical_bytes=" << memory.ullAvailPhys
                  << " available_commit_bytes=" << memory.ullAvailPageFile
                  << " available_virtual_bytes=" << memory.ullAvailVirtual;
    }
    else
        std::cout << " GlobalMemoryStatusEx_error=" << GetLastError();
#endif
    std::cout << std::endl;
    if (context_current) {
        size_t available = 0, total = 0;
        CUresult status = cuMemGetInfo(&available, &total);
        std::cout << "cuMemGetInfo status=" << static_cast<int>(status);
        if (status == CUDA_SUCCESS)
            std::cout << " free_bytes=" << available << " total_bytes=" << total;
        std::cout << std::endl;
    }
}

void
create_pool(Probe &probe, CUmemoryPool &pool)
{
    CUmemPoolProps props{};
    props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
    props.location.type = CU_MEM_LOCATION_TYPE_HOST;
    props.location.id = -1;
    props.maxSize = 2 * allocation_bytes;
    CALL("owned_create", cuMemPoolCreate, &pool, &props);
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CALL("owned_access_grant", cuMemPoolSetAccess, pool, &access, 1);
}

void
destroy_pool(Probe &probe, CUmemoryPool pool, bool revoke)
{
    if (revoke) {
        int count = 0;
        CALL("owned_revoke_device_count", cuDeviceGetCount, &count);
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
        for (int device = 0; device < count; ++device) {
            access.location.id = device;
            CALL("owned_revoke_access", cuMemPoolSetAccess, pool, &access, 1);
        }
    }
    CALL("owned_destroy", cuMemPoolDestroy, pool);
    CALL("owned_post_destroy_sync", cuCtxSynchronize);
}

int
run(const Options &options)
{
    fs::create_directories(options.output);
    // A killed process must not leave an older success report behind.
    fs::remove(options.output / "summary.json");
    fs::remove(options.output / "calls.jsonl");
    Probe probe(options.output);
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUmemoryPool pool = nullptr;
    bool current = false;
    std::string validation_error;
    int driver_version = 0;
    char device_name[256]{};
    std::cout << "native driver-only probe: CUDA headers=" << CUDA_VERSION << " stream=" << options.stream
              << " variant=" << options.variant << " revoke=" << options.revoke << " cycles=" << options.cycles
              << " allocation_bytes=" << allocation_bytes << " requested_max_pool_bytes=" << 2 * allocation_bytes
              << std::endl;
    try {
        CALL("initialize", cuInit, 0);
        CALL("initialize", cuDriverGetVersion, &driver_version);
        CALL("initialize", cuDeviceGetName, device_name, sizeof(device_name), 0);
        std::cout << "driver_api_version=" << driver_version << " device=" << device_name << std::endl;
        CALL("initialize", cuDevicePrimaryCtxRetain, &context, 0);
        CALL("initialize", cuCtxPushCurrent, context);
        current = true;
        memory_snapshot("baseline", current);
        if (options.stream == "nonblocking")
            CALL("private_stream_create", cuStreamCreate, &stream, CU_STREAM_NON_BLOCKING);
        else
            stream = options.stream == "legacy" ? CU_STREAM_LEGACY : CU_STREAM_PER_THREAD;
        CALL("baseline_context_pop", cuCtxPopCurrent, &context);
        current = false;
        for (probe.iteration = 0; probe.iteration < options.cycles; ++probe.iteration) {
            CALL("iteration_context_push", cuCtxPushCurrent, context);
            current = true;
            if (options.variant != "reuse" || probe.iteration == 0)
                create_pool(probe, pool);
            CUdeviceptr pointer = 0;
            CALL("owned_allocate_1MiB", cuMemAllocFromPoolAsync, &pointer, allocation_bytes, pool, stream);
            CALL("owned_free_enqueue", cuMemFreeAsync, pointer, stream);
            CALL("owned_initial_sync", cuCtxSynchronize);
            if (options.variant != "no-trim")
                CALL("owned_trim", cuMemPoolTrimTo, pool, 0);
            if (options.variant != "no-reset") {
                const bool reserved = probe.iteration % 2 == 0;
                auto high = reserved ? CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH : CU_MEMPOOL_ATTR_USED_MEM_HIGH;
                auto used = reserved ? CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT : CU_MEMPOOL_ATTR_USED_MEM_CURRENT;
                CALL("watermark_read_before", cuMemPoolGetAttribute, pool, high, &probe.high_before);
                CALL("watermark_read_before", cuMemPoolGetAttribute, pool, used, &probe.current_before);
                cuuint64_t zero = 0;
                CALL("watermark_set_zero", cuMemPoolSetAttribute, pool, high, &zero);
                CALL("watermark_verify", cuMemPoolGetAttribute, pool, high, &probe.high_after);
                CALL("watermark_verify", cuMemPoolGetAttribute, pool, used, &probe.current_after);
                // Without trimming the allocator may retain backing memory. Only
                // require reset-to-current, not that reserved memory becomes zero.
                if (probe.high_before < allocation_bytes || probe.high_after != probe.current_after)
                    throw std::runtime_error("Unexpected watermark values");
            }
            if (options.variant != "reuse")
                destroy_pool(probe, pool, options.revoke);
            else
                CALL("owned_reuse_sync", cuCtxSynchronize);
            CALL("iteration_context_pop", cuCtxPopCurrent, &context);
            current = false;
            ++probe.completed;
        }
        CALL("cleanup_context_push", cuCtxPushCurrent, context);
        current = true;
        if (options.variant == "reuse")
            destroy_pool(probe, pool, options.revoke);
        if (options.stream == "nonblocking")
            CALL("private_stream_destroy", cuStreamDestroy, stream);
        CALL("cleanup_context_pop", cuCtxPopCurrent, &context);
        current = false;
        CALL("primary_context_release", cuDevicePrimaryCtxRelease, 0);
    }
    catch (const Event &) {
        // Keep the first failing boundary intact; do not retry or run CUDA cleanup.
    }
    catch (const std::runtime_error &error) {
        validation_error = error.what();
    }
    const bool passed = !probe.failed && validation_error.empty();
    probe.dump(options.output);
    {
        std::ofstream summary(options.output / "summary.json");
        summary.exceptions(std::ios::badbit | std::ios::failbit);
        summary << std::boolalpha << "{\"required_path_completed\":" << passed
                << ",\"requested_cycles\":" << options.cycles << ",\"completed_cycles\":" << probe.completed
                << ",\"stream\":\"" << options.stream << "\",\"variant\":\"" << options.variant
                << "\",\"revoke_access\":" << options.revoke << ",\"cuda_header_version\":" << CUDA_VERSION
                << ",\"driver_api_version\":" << driver_version << ",\"failure\":";
        if (probe.failed) {
            const auto &failure = probe.failure;
            summary << "{\"iteration\":" << failure.iteration << ",\"stage\":\"" << failure.stage << "\",\"api\":\""
                    << failure.api << "\",\"status\":" << failure.status << "}";
        }
        else
            summary << "null";
        // Only a fixed, local validation message can reach this field.
        summary << ",\"validation_error\":\"" << validation_error
                << "\",\"last_watermark\":{\"high_before\":" << probe.high_before
                << ",\"current_before\":" << probe.current_before << ",\"high_after\":" << probe.high_after
                << ",\"current_after\":" << probe.current_after << "}}\n";
    }
    std::cout << "completed_cycles=" << probe.completed << " passed=" << passed << std::endl;
    if (!passed) {
        std::cout << "first_failure stage=" << probe.failure.stage << " api=" << probe.failure.api
                  << " status=" << probe.failure.status << " validation_error=" << validation_error << std::endl;
        // Write summary/call history before this optional query, which can itself
        // fail or hang in a poisoned context. It must not replace the first error.
        memory_snapshot("failure", current);
    }
    return passed ? 0 : 1;
}

int
main(int argc, char **argv)
{
    try {
        return run(parse(argc, argv));
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
