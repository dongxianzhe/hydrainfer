import os
import random
from dataclasses import dataclass
from pyverbs.device import get_device_list
import pycuda.autoinit
import pycuda.driver as cuda


@dataclass
class InfinibandDevice:
    name: str
    pci_bus_id: str
    numa_node: int


def get_infiniband_devices() -> list[InfinibandDevice]:
    devices: list[InfinibandDevice] = []
    for device in get_device_list():
        name: str = device.name.decode()
        # Get the PCI bus id for the infiniband device. Note that
        # "/sys/class/infiniband/mlx5_X/" is a symlink to
        # "/sys/devices/pciXXXX:XX/XXXX:XX:XX.X/infiniband/mlx5_X/".
        path = os.path.join(f'/sys/class/infiniband/{device.name.decode()}/../..') 
        path = os.path.realpath(path)
        pci_bus_id = os.path.basename(path)
        with open(os.path.join(path, 'numa_node'), "r") as f:
            numa_node = int(f.read().strip())
        devices.append(InfinibandDevice(name = name, pci_bus_id = pci_bus_id, numa_node = numa_node))
    return devices


@dataclass
class TopologyEntry:
    preferred_hca: list[str] # eg. mlx5_0, mlx5_1
    avail_hca: list[str] # eg. mlx5_2, mlx5_3


def get_cpu_ids() -> list[int]:
    cpu_ids: list[int] = []
    node_dir = "/sys/devices/system/node"
    for entry in os.listdir(node_dir):
        if not entry.startswith("node") or not os.path.isdir(os.path.join(node_dir, entry)):
            continue
        node_id = int(entry[len("node"):])
        cpu_ids.append(node_id)
    return cpu_ids


def get_pci_distance(bus1: str, bus2: str) -> int:
    path1 = os.path.realpath(f"/sys/bus/pci/devices/{bus1.lower()}")
    path2 = os.path.realpath(f"/sys/bus/pci/devices/{bus2.lower()}")
    if not os.path.exists(path1) or not os.path.exists(path2):
        return -1

    i = 0
    while i < min(len(path1), len(path2)) and path1[i] == path2[i]:
        i += 1

    return path1[i:].count('/') + path2[i:].count('/')


def build_topology() -> dict[str, TopologyEntry]:
    all_hca: list[int] = get_infiniband_devices()
    all_cpu: list[int] = get_cpu_ids()
    topology: dict[str, TopologyEntry] = {}
    for cpu_id in all_cpu:
        topology[f'cpu:{cpu_id}'] = TopologyEntry(
            preferred_hca = [hca for hca in all_hca if hca.numa_node == cpu_id], 
            avail_hca = [hca for hca in all_hca if hca.numa_node != cpu_id], 
        ) 
    for i in range(cuda.Device.count()):
        cuda_pci_bus_id = cuda.Device(i).pci_bus_id()
        min_distance = 10000
        preferred_hca: list[str] = []
        for hca in all_hca:
            dist = get_pci_distance(cuda_pci_bus_id, hca.pci_bus_id)
            if dist <= 0:
                continue
            if dist < min_distance:
                min_distance = dist
                preferred_hca = [hca]
            elif dist == min_distance:
                preferred_hca.append(hca)
        avail_hca = [hca for hca in all_hca if hca not in preferred_hca]

        topology[f'cuda:{i}'] = TopologyEntry(
            preferred_hca = preferred_hca, 
            avail_hca = avail_hca, 
        )
    return TopologyGraph(table=topology)

@dataclass
class TopologyGraph:
    table: dict[str, TopologyEntry]

class Topology:
    def __init__(self):
        self.graph = build_topology()
    
    def select_device(self, device: str) -> InfinibandDevice:
        candidates: list[InfinibandDevice] = self.topology.table[device].preferred_hca
        if len(candidates) == 0:
            candidates = self.topology.table[device].avail_hca
        selected: InfinibandDevice = random.choice(candidates)
        return selected