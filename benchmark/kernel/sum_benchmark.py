import torch
import time

start = time.perf_counter()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()

for i in range(100):
    x = torch.rand(4096, 4096, device='cuda')
    y = torch.matmul(x, x)

end_event.record()

torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
end = time.perf_counter()

print(f"Kernel execution time: {elapsed_time} ms")
print(f"Kernel execution time: {(end - start) * 1000} ms")
