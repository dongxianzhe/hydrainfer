import argparse
import subprocess
def get_free_gpus():
    # nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    gpus = result.stdout.splitlines()


    free_gpus = []
    for gpu in gpus:
        index, memory_used, memory_total = map(int, gpu.split(','))
        
        memory_usage_percentage = (memory_used / memory_total) * 100
        
        if memory_usage_percentage < 1:
            free_gpus.append(index)

    return free_gpus

if __name__ == '__main__':
    free_gpus = get_free_gpus()
    print(" ".join(map(str, free_gpus)))