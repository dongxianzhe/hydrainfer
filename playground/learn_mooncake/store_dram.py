import numpy as np
import json
from mooncake.store import MooncakeDistributedStore

from hydrainfer.utils.socket_utils import find_free_port
from abstract import start_mooncake_server
node_port = find_free_port()
master_service_port, mooncake_http_metadata_server_port = start_mooncake_server()

# 1. Initialize
store = MooncakeDistributedStore()
store.setup(f"localhost:{node_port}",
            f"http://localhost:{mooncake_http_metadata_server_port}/metadata",
            512*1024*1024,
            128*1024*1024,
            "tcp",
            "",
            f"localhost:{master_service_port}")
print("Store ready.")

# 2. Store data
store.put("config", b'{"model": "llama-7b", "temperature": 0.7}')
model_weights = np.random.randn(1000, 1000).astype(np.float32)
print(f'model_weights {model_weights.mean()}')

store.put("weights", model_weights.tobytes())
store.put("cache", b"some serialized cache data")

# 3. Retrieve and verify data
config = json.loads(store.get("config").decode())
weights = np.frombuffer(store.get("weights"), dtype=np.float32).reshape(1000, 1000)

print("Config OK:", config["model"])
print("Weights OK, mean =", round(float(weights.mean()), 4))
print("Cache exists?", bool(store.is_exist("cache")))

# 4. Close
store.close()
