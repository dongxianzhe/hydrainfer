import numpy as np
from abstract import Mooncake, MooncakeConfig, make_mooncake_config

# Initialize store
mooncake = Mooncake(make_mooncake_config())
mooncake.start_master_server()
mooncake.start_metadata_server()
store = mooncake.start_store_client()

# Create a large buffer
buffer = np.zeros(100 * 1024 * 1024, dtype=np.uint8)  # 100MB buffer

# Register the buffer for zero-copy operations
buffer_ptr = buffer.ctypes.data
result = store.register_buffer(buffer_ptr, buffer.nbytes)
if result != 0:
    print(f"Failed to register buffer: {result}")
    raise RuntimeError(f"Failed to register buffer: {result}")
print("Buffer registered successfully.")
store.unregister_buffer(buffer_ptr)
