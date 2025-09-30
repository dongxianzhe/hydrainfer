import os
import hydra
import asyncio
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import Literal
from fastapi import Request
import hydrainfer
from hydrainfer.engine import RequestProcessParameters, OutputTokenParams
from hydrainfer.cluster import Cluster, ClusterConfig
from hydrainfer.request import OfflineInferenceOutput
from hydrainfer.entrypoint.api_server import APIServer, APIServerConfig
from hydrainfer.utils.zmq_utils import ZMQConfig, init_zmq_recv
from hydrainfer.utils.socket_utils import parse_network_config
from hydrainfer.utils.counter import Counter
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)


@dataclass
class EntryPointConfig:
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    apiserver: APIServerConfig = field(default_factory=ClusterConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


class EntryPoint:
    def __init__(self, config: EntryPointConfig):
        self.config = config
        self.config.zmq = parse_network_config(self.config.zmq, log_name='zmq')
        self.api_server = APIServer(config.apiserver)
        self.cluster = Cluster(config.cluster)
        self.api_server.register(self.cluster)
        self.api_server.run()

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