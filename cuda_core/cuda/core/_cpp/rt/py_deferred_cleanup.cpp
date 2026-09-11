// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.hpp"
#include "api.hpp"
#include "internal.hpp"
#include <atomic>
#include <stdexcept>

namespace cuda_core::rt {

using namespace detail;

// ============================================================================
// CUDA user-object deferred cleanup
//
// CUDA invokes a user-object destructor on an internal thread where CUDA
// calls are forbidden. Payload cleanup can release resource handles whose
// deleters call CUDA, so the callback only transfers a preallocated intrusive
// node to this process-lifetime queue. One coalesced pending call drains all
// queued payloads from Python's main thread.
// ============================================================================

namespace {
// Process-lifetime MPSC queue that drains payloads from Python's main thread.
class DeferredCleanupQueue {
public:
    // Transfer one preallocated cleanup item from a producer to the queue.
    void enqueue(DeferredCleanupItem* item) noexcept {
        DeferredCleanupItem* head = head_.load(std::memory_order_relaxed);
        do {
            item->next = head;
        } while (!head_.compare_exchange_weak(
            head, item, std::memory_order_release, std::memory_order_relaxed));
        schedule();
    }

    // Permanently disable pending-call scheduling during interpreter shutdown.
    void stop() noexcept {
        accepting_.store(false, std::memory_order_release);
    }

    // Reattempt scheduling after Py_AddPendingCall() found CPython's bounded
    // pending-call queue full and left payloads queued for a later safe entry.
    void retry_schedule() noexcept {
        schedule();
    }

private:
    // Adapt queue draining to CPython's int (*)(void*) callback ABI.
    static int pending_call(void* arg) noexcept {
        static_cast<DeferredCleanupQueue*>(arg)->drain();
        return 0;
    }

    // Coalesce all queued work behind at most one CPython pending call.
    void schedule() noexcept {
        if (!accepting_.load(std::memory_order_acquire)) {
            return;
        }
        if (!Py_IsInitialized() || py_is_finalizing()) {
            stop();
            return;
        }
        if (!head_.load(std::memory_order_acquire)) {
            return;
        }
        bool expected = false;
        if (!scheduled_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return;
        }
        if (Py_AddPendingCall(&DeferredCleanupQueue::pending_call, this) != 0) {
            // Keep every payload queued. A later enqueue or safe cuda-core
            // entry can retry without blocking CUDA's callback thread.
            scheduled_.store(false, std::memory_order_release);
        }
    }

    // Detach and destroy all queued payloads from Python's main thread.
    void drain() noexcept {
        if (!Py_IsInitialized() || py_is_finalizing()) {
            stop();
            scheduled_.store(false, std::memory_order_release);
            return;  // Intentionally leak intact payloads during shutdown.
        }

        while (DeferredCleanupItem* list =
                   head_.exchange(nullptr, std::memory_order_acquire)) {
            while (list) {
                DeferredCleanupItem* next = list->next;
                delete list;
                list = next;
            }
        }

        scheduled_.store(false, std::memory_order_release);
        if (head_.load(std::memory_order_acquire)) {
            schedule();
        }
    }

    // Head of the intrusive multi-producer, single-consumer payload stack.
    std::atomic<DeferredCleanupItem*> head_{nullptr};
    // True while one cuda-core drain callback is pending or executing.
    std::atomic<bool> scheduled_{false};
    // False once shutdown begins, causing later payloads to be leaked safely.
    std::atomic<bool> accepting_{true};
};

// Published once at module initialization and intentionally never freed.
std::atomic<DeferredCleanupQueue*> deferred_cleanup_queue{nullptr};
}  // namespace

namespace detail {
void ensure_deferred_cleanup_ready() {
    DeferredCleanupQueue* queue =
        deferred_cleanup_queue.load(std::memory_order_acquire);
    if (!queue) {
        throw std::runtime_error("deferred cleanup is not initialized");
    }
    queue->retry_schedule();
}

// CUDA's CUhostFn ABI is void (*)(void*); recover and enqueue the cleanup item.
void enqueue_cleanup(void* item) noexcept {
    auto* cleanup = static_cast<DeferredCleanupItem*>(item);
    if (DeferredCleanupQueue* queue =
            deferred_cleanup_queue.load(std::memory_order_acquire)) {
        queue->enqueue(cleanup);
    }
}
}  // namespace detail

// Module initialization calls this once with the GIL held, which serializes
// the check, allocation, and publication below.
void initialize_deferred_cleanup() {
    if (deferred_cleanup_queue.load(std::memory_order_acquire)) {
        return;
    }
    auto* queue = new DeferredCleanupQueue();
    deferred_cleanup_queue.store(queue, std::memory_order_release);
}

void retry_deferred_cleanup() noexcept {
    if (!Py_IsInitialized() || py_is_finalizing()) {
        return;
    }
    if (DeferredCleanupQueue* queue =
            deferred_cleanup_queue.load(std::memory_order_acquire)) {
        queue->retry_schedule();
    }
}

}  // namespace cuda_core::rt
