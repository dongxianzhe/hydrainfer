from dataclasses import dataclass
from hydrainfer.utils.socket_utils import NetworkAddressConfig
from hydrainfer.transfer_engine.topology import Topology


@dataclass
class TransferRequest:
    opcode: int
    source: int
    target_id: int
    target_offset: int
    length: int


class TransferMetadata:
    pass


class Transport:
    pass


class TcpTransport(Transport):
    pass


class NVMeoFTransport(Transport):
    pass


class RdmaTransport(Transport): 
    pass


class NVLinkTransport(Transport):
    pass


class MultiTransport:
    def __init__(self):
        self.transports = {
            "tcp" : TcpTransport(), 
            "nvmeof" :  NVMeoFTransport(), 
            "rdma" : RdmaTransport(), 
            "nvlink" : NVLinkTransport()
        }

    def submit(self, requests: list[TransferRequest]):
        for request in requests:
            
        self.transports


@dataclass
class TransferEngineConfig:
    local_addr: NetworkAddressConfig
    metadata_conn_string: str


class TransferEngine:
    def __init__(self, config: TransferEngineConfig) -> None:
        self.topology = Topology()
        self.metadata = TransferMetadata(config.metadata_conn_string)
        # add rpc meta entry
        # discover


if __name__ == '__main__':
    import pycuda.driver as cuda
    cuda.init()

    device_id = 0
    device = cuda.Device(device_id)

    # CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED 的值是 78（示例，需确认官方文档）
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 78

    from pycuda._driver import device_attribute
    
    val = device.get_attribute(device_attribute.HANDLE_TYPE_FABRIC_SUPPORTED)
    print("Fabric memory support:", val)