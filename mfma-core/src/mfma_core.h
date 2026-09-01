// mfma_core.h — Public interface for MFMA core computation

#ifndef MFMA_CORE_H
#define MFMA_CORE_H

#include <stdint.h>

void mfma_tile_hls_hardware_shim(
    const uint16_t a_tile[256],
    const uint16_t b_tile[256],
    const float    c_tile[256],
    float          out_tile[256]
);

#endif // MFMA_CORE_H
