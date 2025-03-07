#!/bin/bash
export DIR="assets/demo/i2mv/"
export FILE="dino"
export NUMVIEW=6

python -m scripts.inference_i2mv_sdxl \
--filename $FILE \
--image $DIR$FILE.png \
--text "" \
--seed 21 \
--num_views $NUMVIEW \
--device "cuda:0" \
--output output/$FILE-$NUMVIEW.png