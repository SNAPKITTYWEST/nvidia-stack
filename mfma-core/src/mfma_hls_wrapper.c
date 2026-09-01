// mfma_hls_wrapper.c — HLS-compatible C wrapper for MFMA core
// Stripped of Caml runtime allocation in inner hardware-mapped loop

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <stdint.h>
#include <math.h>
#include "mfma_core.h"

// IEEE-754 FP16 to FP32 conversion (matches gfx942 hardware)
static inline float half_to_float_ieee754(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exp == 0x1F) {
        if (mantissa == 0) {
            return sign ? -__builtin_inff() : __builtin_inff();
        } else {
            return __builtin_nanf("");
        }
    } else if (exp == 0) {
        float m = mantissa ? __builtin_ldexpf((float)mantissa, -24) : 0.0f;
        return sign ? -m : m;
    } else {
        float m = __builtin_ldexpf((float)(mantissa | 0x400), (int)exp - 15);
        return sign ? -m : m;
    }
}

// Hardware-mapped MFMA tile shim (HLS pragma-controlled)
void mfma_tile_hls_hardware_shim(
    const uint16_t a_tile[256],
    const uint16_t b_tile[256],
    const float    c_tile[256],
    float          out_tile[256]
) {
    #pragma HLS INTERFACE m_axi port=a_tile bundle=gmem0
    #pragma HLS INTERFACE m_axi port=b_tile bundle=gmem1
    #pragma HLS INTERFACE m_axi port=c_tile bundle=gmem2
    #pragma HLS INTERFACE m_axi port=out_tile bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS PIPELINE II=1

    for (int m = 0; m < 16; m++) {
        for (int n = 0; n < 16; n++) {
            float acc = c_tile[m * 16 + n];
            for (int k = 0; k < 16; k++) {
                #pragma HLS UNROLL
                float va = half_to_float_ieee754(a_tile[m * 16 + k]);
                float vb = half_to_float_ieee754(b_tile[k * 16 + n]);

                // IEEE-754 compliant NaN propagation
                if (__builtin_isnan(va) || __builtin_isnan(vb) || __builtin_isnan(acc)) {
                    acc = __builtin_nanf("");
                } else {
                    acc = __builtin_fma(va, vb, acc);
                }
            }
            out_tile[m * 16 + n] = acc;
        }
    }
}
