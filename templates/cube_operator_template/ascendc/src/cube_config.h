#pragma once
#include <cstdint>

struct CubeConfig {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t batchCount;
    uint32_t blockDim;
    uint32_t matricesPerCore;
    // Tiling parameters
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t depthA;
    uint32_t depthB;
    bool doubleBuffer;
    // Format
    bool transposeA;
    bool transposeB;
    // Dtypes (encoded)
    uint32_t inputDtype;   // 0=FP16, 1=BF16, 2=INT8
    uint32_t accumDtype;   // 0=FP16, 1=FP32
    uint32_t outputDtype;  // 0=FP16, 1=FP32
    // Layout
    uint32_t layoutA;      // 0=ND, 1=NZ
    uint32_t layoutB;
    // Workspace
    uint32_t workspaceSize;
};
