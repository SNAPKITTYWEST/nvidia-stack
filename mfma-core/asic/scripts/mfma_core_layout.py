#!/usr/bin/env python3
# mfma_core_layout.py — GDSFactory Layout Generation for MFMA Core

import gdsfactory as gf

@gf.cell
def mfma_tile_layout():
    c = gf.Component("mfma_tile_gdsii")

    # Core systolic array footprint (16x16x16 FP16 MAC units)
    core = c << gf.components.rectangle(size=(128.0, 128.0), layer=(1, 0))
    core.name = "mfma_systolic_core"

    # Metal 1/Metal 2 power distribution network (PDN) rings
    vdd_ring = c << gf.components.rectangle(size=(132.0, 132.0), layer=(3, 0))
    vdd_ring.center = core.center

    gnd_ring = c << gf.components.rectangle(size=(136.0, 136.0), layer=(4, 0))
    gnd_ring.center = core.center

    return c

if __name__ == "__main__":
    c = mfma_tile_layout()
    c.write_gds("mfma_core.gdsii")
    print("Successfully generated GDSII stream: mfma_core.gdsii")
