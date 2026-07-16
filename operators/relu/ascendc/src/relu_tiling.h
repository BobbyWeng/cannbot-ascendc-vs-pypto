#pragma once
#include <cstdint>

constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t DOUBLE_BUFFER = 2;

struct ReluTilingData {
    uint32_t blockNum;
    uint64_t totalElements;
    uint64_t numPerCore;
    uint64_t tailNumLastCore;
};
