import os
import hydra
import asyncio
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import Literal
from fastapi import Request
import dxz
from dxz.engine import RequestProcessParameters, OutputTokenParams
from dxz.cluster import Cluster, ClusterConfig
from dxz.request import OfflineInferenceOutput
from dxz.entrypoint.api_server import APIServer, APIServerConfig
from dxz.utils.zmq_utils import ZMQConfig, init_zmq_recv
from dxz.utils.counter import Counter


@dataclass
class EntryPointConfig:
    mode: Literal["offline", "online"] = "online"
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    apiserver: APIServerConfig = field(default_factory=ClusterConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


class EntryPoint:
    def __init__(self, config: EntryPointConfig):
        self.config = config
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


@hydra.main(config_path=os.path.join(dxz.__path__[0], 'config'), config_name="entrypoint", version_base=None)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    entrypoint = EntryPoint(config) 
    entrypoint.run()


if __name__ == '__main__':
    main()