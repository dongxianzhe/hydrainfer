import socket
from dataclasses import dataclass
from typing import Optional
from hydrainfer.utils.logger import getLogger
import copy
logger = getLogger(__name__)

@dataclass
class NetworkAddressConfig:
    host: str = "127.0.0.1"
    port: int = -1


def find_free_port() -> int:
    "return a port that is not used"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))
        return s.getsockname()[1]


def parse_port(port_config: int) -> int:
    assert -1 <= port_config <= 65535, f'invalid port {port_config}'
    if port_config == -1:
        return find_free_port()
    return port_config


def parse_host(host_config: str) -> str:
    if host_config == 'auto':
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    else:
        return host_config


def parse_network_config(config: NetworkAddressConfig, log_name: Optional[str] = None) -> NetworkAddressConfig:
    config = copy.deepcopy(config)
    if config.host == 'auto':
        config.host = parse_host(config.host)
        if log_name:
            logger.info(f'auto set {log_name} host to {config.host}')
    if config.port == -1:
        assert -1 <= config.port <= 65535, f'invalid port {config.port}'
        config.port = parse_port(config.port)
        if log_name:
            logger.info(f'auto set {log_name} port to {config.port}')
    return config


def parse_address(config: NetworkAddressConfig) -> str:
    config = parse_network_config(config)
    url = f"tcp://{config.host}:{config.port}"
    return url