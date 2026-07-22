#pragma once
#include <cstdint>

constexpr uint32_t LN_TILE_LEN = 8192;
struct LayerNormTilingData {
    uint32_t blockNum;
    uint64_t totalElements;
    uint32_t rowsPerBlock;
    uint32_t tailRows;
    uint32_t lastDimSize;
    float lastDimSizeF;
    float eps;
};

