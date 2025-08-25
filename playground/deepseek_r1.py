# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="/mnt/cfs/9n-das-admin/llm_models/DeepSeek-R1", trust_remote_code=True)
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)