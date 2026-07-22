#pragma once
#include <cstdint>
struct SoftmaxTilingData {
    uint32_t blockNum;
    uint32_t rowsPerBlock;
    uint32_t tailRows;
    uint32_t lastDimSize;
    float lastDimSizeF;
};
