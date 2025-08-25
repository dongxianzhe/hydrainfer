import os
import socket
import time
import subprocess
import threading
from hydrainfer.utils.socket_utils import find_free_port, NetworkAddressConfig
from dataclasses import dataclass
from mooncake.store import MooncakeDistributedStore


def start_daemon_process(command: list[str], log_file_name: str):
    def _start_daemon_process(command: list[str], log_file_name: str):
        outfile = open(log_file_name, "w")
        process = subprocess.Popen(command, stdout=outfile, stderr=outfile)
        process.wait()
        outfile.close()
    thread = threading.Thread(target=_start_daemon_process, args=(command, log_file_name), daemon=True)
    thread.start()


def wait_server(ip: str, port: int, timeout: int = 30, interval: int = 2):
    """
    Wait for the specified server port to be accessible, indicating that the background process
    has finished initializing.

    :param ip: IP address of the server
    :param port: Port of the server
    :param timeout: Maximum wait time in seconds, default is 30 seconds
    :param interval: Time interval between checks in seconds, default is 2 seconds
    :return: Returns True if the port is accessible; otherwise, returns False
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Create a TCP/IP socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(interval)  # Set timeout for the connection attempt
                result = s.connect_ex((ip, port))  # Try to connect to the server
                
            # If result == 0, it means the connection was successful
            if result == 0:
                print(f"Port {port} is accessible, service has initialized.")
                return True
        except socket.error as e:
            print(f"Unable to connect to {ip}:{port}, error: {e}")
        
        # Wait for a while before trying again
        print(f"Waiting for {interval} seconds, retrying...")
        time.sleep(interval)
    
    print(f"Port {port} was not accessible within {timeout} seconds, service not initialized.")
    return False


def start_mooncake_server() -> list[int, int]:
    master_service_port: int = find_free_port()
    mooncake_http_metadata_server_port: int = find_free_port()
    print(f'master_service_port {master_service_port}')
    print(f'mooncake_http_metadata_server_port {mooncake_http_metadata_server_port}')
    start_daemon_process(['mooncake_master', '--port', str(master_service_port)], 'mooncake_master.log')
    start_daemon_process(['mooncake_http_metadata_server', '--port', str(mooncake_http_metadata_server_port)], 'mooncake_http_metadata_server.log')
    wait_server(ip='127.0.0.1', port=master_service_port)
    wait_server(ip='127.0.0.1', port=mooncake_http_metadata_server_port)
    return master_service_port, mooncake_http_metadata_server_port


@dataclass
class MooncakeConfig:
    master_server: NetworkAddressConfig
    metadata_server: NetworkAddressConfig
    store_client: NetworkAddressConfig

class Mooncake:
    def __init__(self, config: MooncakeConfig):
        self.config = config
        self.log_directory = "mooncake_log"
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

    def start_master_server(self):
        start_daemon_process(['mooncake_master', '-rpc_address', self.config.master_server.host ,'-rpc_port', str(self.config.master_server.port)], 'mooncake_log/mooncake_master.log')
        wait_server(ip=self.config.master_server.host, port=self.config.master_server.port)
        pass

    def start_metadata_server(self):
        start_daemon_process(['mooncake_http_metadata_server', '--host', self.config.metadata_server.host,'--port', str(self.config.metadata_server.port)], 'mooncake_log/mooncake_http_metadata_server.log')
        wait_server(ip=self.config.metadata_server.host, port=self.config.metadata_server.port)

    def start_store_client(self) -> MooncakeDistributedStore:
        store = MooncakeDistributedStore()
        store.setup(f"{self.config.store_client.host}:{self.config.store_client.port}",
                    f"http://{self.config.metadata_server.host}:{self.config.metadata_server.port}/metadata",
                    512 * 1024 * 1024,
                    128 * 1024 * 1024,
                    "tcp",
                    "",
                    f"{self.config.master_server.host}:{self.config.master_server.port}")
        return store

def make_mooncake_config() -> MooncakeConfig:
    return MooncakeConfig(
        master_server =  NetworkAddressConfig(
            host = "127.0.0.1", 
            port = find_free_port(), 
        ), 
        metadata_server = NetworkAddressConfig(
            host = "127.0.0.1", 
            port = find_free_port(), 
        ), 
        store_client = NetworkAddressConfig(
            host = "127.0.0.1", 
            port = find_free_port(), 
        ), 
    )
