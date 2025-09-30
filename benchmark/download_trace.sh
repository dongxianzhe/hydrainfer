#!/usr/bin/env bash

DATASET="/datasets"
mkdir -p $DATASET

cd $DATASET
# Prepare prompts
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Prepare traces
# Download the "Azure LLM Inference Traces"
wget https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_code.csv
wget https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_conv.csv

# Download the "BurstGPT" traces
wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_1.csv
wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv

# Concat two files
head -n 1 BurstGPT_without_fails_1.csv > BurstGPT_without_fails.csv && tail -n +2 -q BurstGPT_without_fails_1.csv BurstGPT_without_fails_2.csv >> BurstGPT_without_fails.csv
rm BurstGPT_without_fails_1.csv BurstGPT_without_fails_2.csv

# Download the "Mooncake" traces
wget https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/conversation_trace.jsonl -O mooncake_conv_trace.jsonl
