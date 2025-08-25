import subprocess
import time
from mooncake.store import MooncakeDistributedStore
from hydrainfer.utils.socket_utils import find_free_port
from abstract import start_mooncake_server

master_service_port, mooncake_http_metadata_server_port = start_mooncake_server()
node_port: int = find_free_port()

print(f'node_port {node_port}')

# 1. Create store instance
store = MooncakeDistributedStore()

# 2. Setup with all required parameters
store.setup(
    f"localhost:{node_port}",           # Your node's address
    f"http://localhost:{mooncake_http_metadata_server_port}/metadata",    # HTTP metadata server
    512*1024*1024,          # 512MB segment size
    128*1024*1024,          # 128MB local buffer
    "tcp",                             # Use TCP (RDMA for high performance)
    "",                            # Empty for TCP, specify device for RDMA
    f"localhost:{master_service_port}"        # Master service
)

# 3. Store data
store.put("hello_key", b"Hello, Mooncake Store!")

# 4. Retrieve data
data = store.get("hello_key")
print('##############################')
print(data)
print(data.decode())  # Output: Hello, Mooncake Store!
print('##############################')

# 5. Clean up
store.close()