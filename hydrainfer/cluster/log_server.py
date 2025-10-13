import json
import ray
from dataclasses import dataclass, asdict
import asyncio
import ray.actor
from hydrainfer.utils.zmq_utils import ZMQConfig, init_zmq_recv, init_zmq_send
from hydrainfer.utils.logger import getLogger
from hydrainfer.utils.socket_utils import parse_network_config
logger = getLogger(__file__)


@dataclass
class LatencyBreakDownMetric:
    request_id: str
    name: str
    time_stamp: float


@dataclass
class LogServerConfig:
    log_path: str = "hydrainfer_log_server.json"


@ray.remote
class LogServer:
    def __init__(self, config: LogServerConfig):
        self.config = config
        self.zmq_config = ZMQConfig(host='auto', port=-1)
        self.zmq_config = parse_network_config(self.zmq_config, log_name='log server')
        self.zmq_recv = init_zmq_recv(self.zmq_config)
        asyncio.create_task(self._zmq_recv_loop())
        logger.info(f'log server is start on {self.zmq_config}')
        self.file = open(self.config.log_path, 'w')

    async def _zmq_recv_loop(self):
        while True:
            log = await self.zmq_recv.recv_pyobj()
            self.file.write(json.dumps(asdict(log)) + "\n")
            self.file.flush()
            await asyncio.sleep(0)

    async def get_zmq_config(self) -> ZMQConfig:
        return self.zmq_config
        