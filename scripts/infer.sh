#!/bin/bash
export DIR="assets/demo/i2mv/"
export FILE="dslr"
export NUMVIEW=16

python -m scripts.inference_i2mv_sdxl \
--image $DIR$FILE.png \
--text "" \
--seed 21 \
--num_views $NUMVIEW \
--output out_$FILE_$NUMVIEW.png