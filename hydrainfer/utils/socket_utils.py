import socket
import psutil
from dataclasses import dataclass
from typing import Optional
from hydrainfer.utils.logger import getLogger
import copy
logger = getLogger(__name__)

@dataclass
class NetworkAddressConfig:
    host: str = "127.0.0.1"
    port: int = -1

_free_port_used = set()
def find_free_port() -> int:
    "return a port that is not used"
    for _ in range(100000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', 0))
            port = s.getsockname()[1]
            if not port in _free_port_used:
                _free_port_used.add(port)
                return port
    raise Exception('find free port failed')


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


@dataclass
class snicaddr:
    """
        eg. snicaddr(family=<AddressFamily.AF_INET: 2>, address='127.0.0.1', netmask='255.0.0.0', broadcast=None, ptp=None)
    """
    family: int
    address: str
    netmask: str
    broadcast: str
    

def find_interface_by_ip(target_ip: str) -> Optional[str]:
    addrs: dict[str, list[snicaddr]] = psutil.net_if_addrs()
    
    for interface, address_list in addrs.items():
        for address in address_list:
            if address.family == socket.AF_INET and address.address == target_ip:
                return interface
    return None