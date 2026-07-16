#pragma once
#include <cstdint>
constexpr uint32_t DOUBLE_BUFFER = 2;
struct TransposeTilingData {
    uint32_t blockNum;
    uint64_t totalBatches;
    uint32_t H;
    uint32_t W;
    uint32_t tileH;
    uint32_t tileW;
};
