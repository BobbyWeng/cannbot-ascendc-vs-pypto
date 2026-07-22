#pragma once
#include <cstdint>

struct LayerNormTilingData {
    uint32_t blockNum;
    uint64_t totalElements;
    uint32_t rowsPerBlock;
    uint32_t tailRows;
    uint32_t lastDimSize;       // 32
    float lastDimSizeF;         // 32.0f
    float eps;                  // 1e-5f
};
