#!/usr/bin/env bash

export PROJECT_PATH=workspace/VQ-VAE-Transformer-Arc-Welding/

docker run \
    --gpus=all \
    --interactive \
    --tty \
    --volume "$HOME/$PROJECT_PATH:/home/mambauser/dev-container" \
    --user root \
    VQ-VAE-Transformer-Arc-Welding \
    bash
