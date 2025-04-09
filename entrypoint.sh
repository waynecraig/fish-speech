#!/bin/bash

CUDA_ENABLED=${CUDA_ENABLED:-true}
DEVICE=""

if [ "${CUDA_ENABLED}" != "true" ]; then
    DEVICE="--device cpu"
fi

exec python tools/run_webui2.py ${DEVICE} --llama-checkpoint-path checkpoints/fish-speech-1.5-wh8
