#!/bin/bash
export DIR="assets/demo/i2mv/"
export FILE="dino"
export NUMVIEW=4

python -m scripts.inference_i2mv_sdxl \
--image $DIR$FILE.png \
--text "" \
--seed 21 \
--num_views $NUMVIEW \
--device "cuda:0" \
--output out_${FILE}_$NUMVIEW.png