import subprocess
import sys, time
import ray
from dxz.utils.logger import getLogger
logger = getLogger(__name__)


def get_ip_address() -> str:
    result = subprocess.run(['hostname', '-i'], stdout=subprocess.PIPE, check=True)
    ip_address = result.stdout.decode('utf-8').strip()
    return ip_address


def stop_ray_process():
    try:
        logger.info('try to stop ray cluster')
        subprocess.run(['ray', 'stop', '--force'], check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error("'ray stop' failed with: \n{}".format(e.stderr))
        sys.exit(1)


def start_head_node(ray_cluster_port: int):
    node_ip_address = get_ip_address()
    try:
        logger.info('try to start ray head node')
        result = subprocess.run(['ray', 'start', '--head', f'--port={ray_cluster_port}'], check=True, text=True, capture_output=True)
        logger.info('successfully start ray head node')
    except subprocess.CalledProcessError as e:
        ray_start_command = f"ray start --head --node-ip-address={node_ip_address} --port={ray_cluster_port}"
        logger.error("'{}' failed with: \n{}".format(ray_start_command, e.stderr))
        sys.exit(1)


def start_worker_node(head_node_ip: str, ray_cluster_port: int, max_restarts: int = 30, restart_interval: float = 1.):
    node_ip_address = get_ip_address()
    for attempt in range(max_restarts):
        try:
            logger.info('try to connect to ray head node')
            # wait about 2 mins by default
            result = subprocess.run(['ray', 'start', f'--address={head_node_ip}:{ray_cluster_port}'], check=True, text=True, capture_output=True)
            logger.info('successfully connect to ray head node')
        except subprocess.CalledProcessError as e:
            if attempt < max_restarts:
                ray_start_command = f"ray start --address={head_node_ip}: {ray_cluster_port} --node-ip-address={node_ip_address}"
                logger.info("execute '{}' repeatedly until the head node starts...".format(ray_start_command))
                time.sleep(restart_interval)
            else:
                logger.error("'{}' failed after {} attempts with: \n{}".format(ray_start_command, attempt, e.stderr))
                sys.exit(1)
        

def launch_ray_cluster(is_head_node: bool, head_node_ip: str, ray_cluster_port: int, namespace: str) -> None:
    stop_ray_process()
    if is_head_node:
        start_head_node(ray_cluster_port)        
    else:
        start_worker_node(head_node_ip, ray_cluster_port)
    ray.init(address=f"{head_node_ip}:{ray_cluster_port}", ignore_reinit_error=True, namespace=namespace)