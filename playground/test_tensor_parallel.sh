#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
RESULT_PATH=$(echo "$SCRIPT_PATH/result/$(date +%Y%m%d_%H%M%S)")

mkdir -p $RESULT_PATH
RAY_DEDUP_LOGS=0 conda run -n hydrainfer --no-capture-output python test_tensor_parallel.py
sleep 10
cd /tmp/ray/session_latest/logs/nsight
cp *.nsys-rep $RESULT_PATH