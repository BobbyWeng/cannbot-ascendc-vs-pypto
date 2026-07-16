#pragma once
#include <cstdint>
constexpr uint32_t DOUBLE_BUFFER = 2;
struct ReduceSumTilingData {
    uint32_t blockNum;
    uint64_t totalRows;    // B * 256
    uint32_t reduceLen;    // 384
};
