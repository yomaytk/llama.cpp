#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

struct ggml_webgpu_binding_key {
    uint32_t binding;
    uintptr_t buffer;
    uint64_t offset;
    uint64_t size;

    bool operator==(const ggml_webgpu_binding_key & other) const {
        return binding == other.binding && buffer == other.buffer && offset == other.offset && size == other.size;
    }
};

struct ggml_webgpu_bind_group_key {
    uintptr_t pipeline;
    std::vector<ggml_webgpu_binding_key> bindings;

    bool operator==(const ggml_webgpu_bind_group_key & other) const {
        return pipeline == other.pipeline && bindings == other.bindings;
    }
};

struct ggml_webgpu_bind_group_hash {
    size_t operator()(const ggml_webgpu_bind_group_key & key) const {
        size_t seed = std::hash<uintptr_t>{}(key.pipeline);
        const auto mix = [&seed](size_t value) { seed ^= value + 0x9e3779b9u + (seed << 6) + (seed >> 2); };
        for (const auto & binding : key.bindings) {
            mix(std::hash<uint32_t>{}(binding.binding));
            mix(std::hash<uintptr_t>{}(binding.buffer));
            mix(std::hash<uint64_t>{}(binding.offset));
            mix(std::hash<uint64_t>{}(binding.size));
        }
        return seed;
    }
};

// Values retain the native handles represented by each key until eviction.
template <typename Value, typename Hash = ggml_webgpu_bind_group_hash>
class ggml_webgpu_bind_group_cache {
    std::unordered_map<ggml_webgpu_bind_group_key, Value, Hash> entries;
    size_t capacity;

public:
    explicit ggml_webgpu_bind_group_cache(size_t capacity = 4096) : capacity(capacity) {}

    const Value * find(const ggml_webgpu_bind_group_key & key) const {
        const auto it = entries.find(key);
        return it == entries.end() ? nullptr : &it->second;
    }

    void insert(ggml_webgpu_bind_group_key key, Value value) {
        if (capacity == 0) { return; }
        if (entries.size() >= capacity) { entries.clear(); }
        entries.emplace(std::move(key), std::move(value));
    }

    void clear() { entries.clear(); }
    size_t size() const { return entries.size(); }
};
