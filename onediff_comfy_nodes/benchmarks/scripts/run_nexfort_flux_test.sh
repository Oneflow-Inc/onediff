#!/bin/bash
set -e

COMFY_PORT=10096
export WORKFLOW_DIR=resources/workflows/nexfort
export WORKFLOW_BASIC=resources/workflows/baseline

# Run the FLUX1 baseline workflow
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_BASIC/flux1-schnell.json \
    --output-dir results \
    --exp-name flux1-schnell-torch \
    --output-images

# Run the FLUX1 baseline workflow
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_BASIC/flux1-schnell.json \
    --output-dir results \
    --exp-name flux1-schnell-baseline \
    --output-images


# Run the FLUX1 nexfort workflow
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/flux1-schnell.json \
    --output-dir results \
    --exp-name flux1-schnell_2_3 \
    --output-images \
    --baseline-dir results/flux1-schnell-baseline

# Run the FLUX1 nexfort workflow
python3 scripts/text_to_image.py \
    --comfy-port $COMFY_PORT \
    -w $WORKFLOW_DIR/flux1-schnell.json \
    --output-dir results \
    --exp-name flux1-schnell_2_3 \
    --output-images \
    --baseline-dir results/flux1-schnell-baseline
