#pragma once
#include <cstdint>

constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t X2_PER_TILE = TILE_LENGTH / 256;
constexpr uint32_t X2_DIVISOR = 256;

struct DivTilingData {
    uint32_t blockNum;
    uint64_t totalElements;
    uint64_t numPerCore;
    uint64_t tailNumLastCore;
};
