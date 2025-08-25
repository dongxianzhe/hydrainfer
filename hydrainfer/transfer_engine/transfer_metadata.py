import json
from dataclasses import dataclass, asdict, fields
from typing import Any
from hydrainfer.transfer_engine.topology import Topology, TopologyEntry, TopologyGraph
from hydrainfer.transfer_engine.storage_plugin import HTTPStoragePlugin

@dataclass
class DeviceDesc:
    name: str
    lid: int
    gid: str

@dataclass
class BufferDesc: 
    name: str
    addr: int
    length: int
    lkey: int # rdma
    rkey: int # rdma
    shm_name: str # nvlink

@dataclass
class NVMeoFBufferDesc:
    file_path: str
    length: int
    local_path_map: dict[str, str]

@dataclass
class SegmentDesc:
    name: str
    protocol: str
    # this is for rdma/shm
    devices: list[DeviceDesc]
    topology_graph: TopologyGraph
    buffers: list[BufferDesc]
    # this is for nvmeof.
    nvmeof_buffers: list[NVMeoFBufferDesc]
    timestamp: str
    tcp_data_port: int

@dataclass
class RpcMetaDesc:
    host: str
    port: int
    sockfd: int # local cache

def dict_to_dataclass(cls: Any, data: dict) -> Any:
    for field in fields(cls):
        if isinstance(data.get(field.name), dict) and hasattr(field.type, '__dataclass_fields__'):
            data[field.name] = dict_to_dataclass(field.type, data[field.name])
    return cls(**data)


def from_json(json_str: str, dataclass_type: Any):
    data = json.loads(json_str)
    
    for field in dataclass_type.__dataclass_fields__:
        field_type = dataclass_type.__dataclass_fields__[field].type
        if hasattr(field_type, '__dataclass_fields__'):
            data[field] = from_json(json.dumps(data[field]), field_type)
    
    return dataclass_type(**data)

class TransferMetadata:
    def __init__(self, conn_str: str):
        self.storage_plugin = HTTPStoragePlugin(metadata_uri=conn_str)
    
    def updateSegmentDesc(self, segment_name: str, desc: SegmentDesc):
        self.storage_plugin.set(segment_name, json.dumps(asdict(desc)))

    def removeSegmentDesc(self, segment_name: str):
        self.storage_plugin.remove(segment_name)

    def getSegmentDesc(self, segment_name: str):
        return self.storage_plugin.get(segment_name)[1]



if __name__ == '__main__':
    desc = SegmentDesc(
        name = "segname", 
        protocol = "nvlink", 
        devices = [DeviceDesc(
            name='cpu:1', 
            lid=0, 
            gid=1, 
        )], 
        topology_graph = Topology().graph, 
        buffers = [BufferDesc(
            name = "buffer0", 
            addr = 2, 
            length = 3, 
            lkey = 4, 
            rkey = 5, 
            shm_name = 6, 
        )], 
        nvmeof_buffers = [NVMeoFBufferDesc(
            file_path = "file.pt", 
            length = 7, 
            local_path_map = {
                "a" : "b", 
            }, 
        )], 
        timestamp = "2025", 
        tcp_data_port = 12345, 
    )

    metadata = TransferMetadata(f'http://localhost:8082/metadata')
    metadata.updateSegmentDesc("test", desc)
    value = metadata.getSegmentDesc("test")
    print(type(value))
    desc = dict_to_dataclass(SegmentDesc, value)
    print(desc)
    

    