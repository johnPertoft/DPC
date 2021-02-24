#!/usr/bin/env bash
set -e
docker build -t dpc .
docker run -it --rm --gpus all --ipc host -v $(pwd):/workspace dpc