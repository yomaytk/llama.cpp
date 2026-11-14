#pragma once

#include <cstdint>

// The caller holds the device queue mutex while changing this state.
struct ggml_webgpu_queue_state {
    uint64_t submitted_serial = 0;
    uint64_t read_serial = UINT64_MAX;

    void submitted() { ++submitted_serial; }
    void read_completed() { read_serial = submitted_serial; }
    bool covers(uint64_t serial) const { return read_serial != UINT64_MAX && read_serial >= serial; }
    bool needs_wait() const { return !covers(submitted_serial); }
};

// An event records a position on its device queue, not a CPU completion promise.
struct ggml_webgpu_event_state {
    uint64_t serial = 0;
    bool recorded = false;

    void record(const ggml_webgpu_queue_state & queue) {
        serial = queue.submitted_serial;
        recorded = true;
    }
    void clear() { recorded = false; }
    bool needs_host_wait(const ggml_webgpu_queue_state & queue) const {
        return recorded && !queue.covers(serial);
    }
    bool needs_backend_wait(const ggml_webgpu_queue_state & queue, bool same_queue) const {
        return !same_queue && needs_host_wait(queue);
    }
};
