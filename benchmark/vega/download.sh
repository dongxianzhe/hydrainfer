#!/bin/bash

curl -L -o IITC_4k_test.json https://huggingface.co/datasets/zhourax977/VEGA/resolve/main/datas/IITC_4k_test.json

curl -L -o imgs.zip https://huggingface.co/datasets/zhourax977/VEGA/resolve/main/imgs.zip

# huggingface-cli download zhourax977/VEGA datas/IITC_4k_test.json --repo-type dataset --local-dir ./
# huggingface-cli download zhourax977/VEGA imgs.zip --repo-type dataset --local-dir ./

echo "vega dataset download finished"