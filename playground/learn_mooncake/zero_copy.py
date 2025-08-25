import numpy as np
from mooncake.store import MooncakeDistributedStore
from abstract import make_mooncake_config, Mooncake

mooncake = Mooncake(make_mooncake_config())
mooncake.start_master_server()
mooncake.start_metadata_server()

# Initialize store with RDMA protocol for maximum performance
store = mooncake.start_store_client()

# Create data to store
original_data = np.random.randn(1000, 1000).astype(np.float32)
buffer_ptr = original_data.ctypes.data
size = original_data.nbytes

# Step 1: Register the buffer
result = store.register_buffer(buffer_ptr, size)
if result != 0:
    raise RuntimeError(f"Failed to register buffer: {result}")

# Step 2: Zero-copy store
result = store.put_from("large_tensor", buffer_ptr, size)
if result == 0:
    print(f"Successfully stored {size} bytes with zero-copy")
else:
    raise RuntimeError(f"Store failed with code: {result}")

# Step 3: Pre-allocate buffer for retrieval
retrieved_data = np.empty((1000, 1000), dtype=np.float32)
recv_buffer_ptr = retrieved_data.ctypes.data
recv_size = retrieved_data.nbytes

# Step 4: Register receive buffer
result = store.register_buffer(recv_buffer_ptr, recv_size)
if result != 0:
    raise RuntimeError(f"Failed to register receive buffer: {result}")

# Step 5: Zero-copy retrieval
bytes_read = store.get_into("large_tensor", recv_buffer_ptr, recv_size)
if bytes_read > 0:
    print(f"Successfully retrieved {bytes_read} bytes with zero-copy")
    # Verify the data
    print(f"Data matches: {np.array_equal(original_data, retrieved_data)}")
else:
    raise RuntimeError(f"Retrieval failed with code: {bytes_read}")

# Step 6: Clean up - unregister both buffers
# store.unregister_buffer(buffer_ptr)
# store.unregister_buffer(recv_buffer_ptr)
store.close()