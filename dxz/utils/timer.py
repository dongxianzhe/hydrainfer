import time
from contextlib import contextmanager

@contextmanager
def timer(context_name: str):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f'{context_name} took {end_time - start_time:.2}s')