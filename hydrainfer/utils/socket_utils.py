import socket
from dataclasses import dataclass


@dataclass
class NetworkAddressConfig:
    host: str = "127.0.0.1"
    port: int = -1


def find_free_port():
    "return a port that is not used"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))
        return s.getsockname()[1]


def parse_port(port_config: int) -> int:
    if port_config == -1:
        return find_free_port()
    return port_config


def parse_address(config: NetworkAddressConfig):
    port = parse_port(config.port)
    assert 0 <= port <= 65535, f'invalid port {port}'
    url = f"tcp://{config.host}:{port}"
    return url