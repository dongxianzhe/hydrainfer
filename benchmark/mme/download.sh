#!/bin/bash

curl -L -o test-00000-of-00004-a25dbe3b44c4fda6.parquet "https://huggingface.co/datasets/lmms-lab/MME/resolve/main/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet"
curl -L -o test-00001-of-00004-7d22c7f1aba6fca4.parquet "https://huggingface.co/datasets/lmms-lab/MME/resolve/main/data/test-00001-of-00004-7d22c7f1aba6fca4.parquet"
curl -L -o test-00002-of-00004-594798fd3f5b029c.parquet "https://huggingface.co/datasets/lmms-lab/MME/resolve/main/data/test-00002-of-00004-594798fd3f5b029c.parquet"
curl -L -o test-00003-of-00004-53ae1794f93b1e35.parquet "https://huggingface.co/datasets/lmms-lab/MME/resolve/main/data/test-00003-of-00004-53ae1794f93b1e35.parquet"

# huggingface-cli download lmms-lab/MME data/test-00000-of-00004-a25dbe3b44c4fda6.parquet --repo-type dataset --local-dir ./
# huggingface-cli download lmms-lab/MME data/test-00001-of-00004-a25dbe3b44c4fda6.parquet --repo-type dataset --local-dir ./
# huggingface-cli download lmms-lab/MME data/test-00002-of-00004-a25dbe3b44c4fda6.parquet --repo-type dataset --local-dir ./
# huggingface-cli download lmms-lab/MME data/test-00003-of-00004-a25dbe3b44c4fda6.parquet --repo-type dataset --local-dir ./

echo "mme dataset download finished"