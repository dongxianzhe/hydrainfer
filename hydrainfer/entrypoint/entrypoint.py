import os
import hydra
import asyncio
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import Literal
from fastapi import Request
import hydrainfer
from hydrainfer.engine import RequestProcessParameters, OutputTokenParams
from hydrainfer.cluster import Cluster, ClusterConfig, NCCLCommunicatorConfig
from hydrainfer.request import OfflineInferenceOutput
from hydrainfer.entrypoint.api_server import APIServer, APIServerConfig
from hydrainfer.utils.zmq_utils import ZMQConfig, init_zmq_recv
from hydrainfer.utils.socket_utils import parse_network_config
from hydrainfer.utils.counter import Counter
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)


@dataclass
class EntryPointConfig:
    mode: Literal["offline", "online"] = "online"
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    nccl_communicator: NCCLCommunicatorConfig = field(default_factory=NCCLCommunicatorConfig)
    apiserver: APIServerConfig = field(default_factory=ClusterConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


class EntryPoint:
    def __init__(self, config: EntryPointConfig):
        self.config = config
        self.config.zmq = parse_network_config(self.config.zmq, log_name='zmq')
        self.config.nccl_communicator = parse_network_config(self.config.nccl_communicator, log_name='nccl_communicator')
        self.cluster = Cluster(config.cluster)
        if self.config.mode == 'online':
            self.api_server = APIServer(config.apiserver)
            self.api_server.register(self.cluster)
            self.api_server.run()
        elif self.config.mode == 'offline':
            self.zmq_recv = init_zmq_recv(self.config.zmq)
        else:
            raise Exception(f'entrypoint invalid mode {self.config.mode}')

    async def _generate(self, requests: list[Request]) -> list[OfflineInferenceOutput]:
        finished = Counter()
        for request in requests:
            self.cluster.add_request(request, RequestProcessParameters(
                outout_token_parmas = OutputTokenParams(
                    print_output_text = False, 
                    is_stream_output = False, 
                    is_offline_output = True
                )
            ))
        outputs: list[OfflineInferenceOutput] = []
        while finished.value() < len(requests):
            output = await self.zmq_recv.recv_pyobj()
            outputs.append(output)
            finished.count()
        return outputs

    def generate(self, requests: list[Request]) -> list[OfflineInferenceOutput]:
        if self._is_event_loop_running():
            loop = asyncio.get_event_loop()
            task = loop.create_task(self._generate(requests))
            outputs = loop.run_until_complete(task)
        else:
            outputs = asyncio.run(self._generate(requests))
        return outputs

    def _is_event_loop_running(self):
        try:
            loop = asyncio.get_event_loop()
            return loop.is_running()
        except RuntimeError:
            return False


@hydra.main(config_path=os.path.join(hydrainfer.__path__[0], 'config'), config_name="entrypoint", version_base=None)
def main(config: DictConfig):
    logger.info(OmegaConf.to_yaml(config))
    entrypoint = EntryPoint(config) 


if __name__ == '__main__':
    main()