#!/bin/bash
set -e

conda run -n base --no-capture-output python create.py --python=3.12 --package=vllm --versions 0.11.0 0.10.2 0.9.2 0.8.5 0.7.3 0.6.6