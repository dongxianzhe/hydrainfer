import requests
import json
import concurrent.futures
import time
import numpy as np
import tqdm

# url = "http://localhost:8888/generate"
url = "http://localhost:9999/v1/completions"

headers = {
    "Content-Type": "application/json"
}

def request():
    start_time = time.time()

    data = {
        "model": "gpt2",
        "prompt":"Hello",
        "max_tokens": 50,
        "temperature": 0
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")

    end_time = time.time()
    end_to_end_time = end_time - start_time
    return end_to_end_time

if __name__ == '__main__':
    # 1. parameters
    concurrency = 10
    total_requests = 100

    # 2. start stress test
    response_times: list[float] = []
    start_time = time.time()
    response_times = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(request) for _ in range(total_requests)]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures)):
            response_time = future.result()
            response_times.append(response_time)
    end_time = time.time()
    # 3. stress test result
    total_time_taken = end_time - start_time
    average_time     = sum(response_times) / len(response_times)
    max_time         = max(response_times)
    tp99_time        = np.percentile(response_times, 99)
    qps              = total_requests / total_time_taken

    print(f'total_time_taken: {total_time_taken}')
    print(f'max_time        : {max_time        }')
    print(f'average_time    : {average_time    }')
    print(f'tp99_time       : {tp99_time       }')
    print(f'qps             : {qps             }')