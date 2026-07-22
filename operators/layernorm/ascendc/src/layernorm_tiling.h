#pragma once
#include <cstdint>

constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t DOUBLE_BUFFER = 2;

struct LayerNormTilingData {
    uint32_t blockNum;
    uint64_t totalElements;
    uint64_t numPerCore;
    uint64_t tailNumLastCore;
    uint32_t lastDimSize;       // size of the last dimension (32)
    float lastDimSizeF;         // lastDimSize as float for aicore use
    float eps;                  // epsilon for numerical stability
};
