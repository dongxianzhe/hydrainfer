#!/bin/bash

curl -L -o test-00000-of-00002.parquet "https://huggingface.co/datasets/lmms-lab/TextCaps/resolve/main/data/test-00000-of-00002.parquet"
curl -L -o test-00001-of-00002.parquet "https://huggingface.co/datasets/lmms-lab/TextCaps/resolve/main/data/test-00001-of-00002.parquet"

# huggingface-cli download lmms-lab/TextCaps data/test-00000-of-00002.parquet --repo-type dataset --local-dir ./
# huggingface-cli download lmms-lab/TextCaps data/test-00001-of-00002.parquet --repo-type dataset --local-dir ./

echo "textcaps dataset download finished"
