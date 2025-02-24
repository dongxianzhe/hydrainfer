import ray
import asyncio
import uvicorn
import argparse
from typing import Literal
from dataclasses import dataclass, field, fields
from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
from dxz.model import ModelFactoryConfig
from dxz.engine import RequestProcessParameters, OutputTokenParams
from dxz.cluster import NodeConfig, AsyncEPDNode, LoadBalancer, LoadBalancerConfig, Cluster, ClusterConfig
from dxz.request import OfflineInferenceOutput
from dxz.entrypoint.api_server import APIServer
from dxz.utils.config_util import CLIConfig
from dxz.utils.zmq_utils import init_zmq_recv
from dxz.utils.ray_utils import get_ip_address
from dxz.utils.counter import Counter



@dataclass
class EntryPointConfig(CLIConfig):
    host: str = '127.0.0.1'
    port: int = 8888
    cluster_config: ClusterConfig = field(default_factory=ClusterConfig)


class EntryPoint:
    def __init__(self, config: EntryPointConfig):
        self.config = config
        self.cluster = Cluster(config.cluster_config)
        self.zmq_recv = self.cluster.get_zmq_recv()
        self.api_server = APIServer(self.zmq_recv)
        self.api_server.register(self.cluster)

    def run(self):
        self.api_server.run(self.config.host, self.config.port)

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


if __name__ == '__main__':
    from PIL import Image
    from dxz.request import SamplingParameters
    from dxz.request import Request
    requests = [
        Request(
            request_id = i, 
            prompt = f"<image>What is the content of this image?", 
            image = Image.open('./benchmark/dataset/cherry_blossom.jpg'), 
            image_base64 = None, 
            sampling_params = SamplingParameters(max_tokens=100)
        )
        for i in range(2)
    ]
    parser = argparse.ArgumentParser(description="entrypoint", conflict_handler='resolve')
    parser = EntryPointConfig.add_cli_args(parser)
    args = parser.parse_args()
    config = EntryPointConfig.from_cli_args(args)
    print(f'config {config}')
    entrypoint = EntryPoint(config) 
    entrypoint.run()