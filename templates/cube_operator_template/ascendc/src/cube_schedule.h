#pragma once
#include <cstdint>

struct CubeSchedule {
    uint32_t stepM;
    uint32_t stepN;
    uint32_t blockDim;
    uint32_t matricesPerCore;
    uint32_t tailMatrices;
    bool doubleBufferEnabled;
};
