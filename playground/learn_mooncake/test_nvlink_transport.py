import os
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化 CUDA 驱动

def support_fabric_mem():
    # 检查环境变量 MC_USE_NVLINK_IPC
    if os.getenv("MC_USE_NVLINK_IPC"):
        return False

    # 获取设备数量
    num_devices = cuda.Device.count()
    if num_devices == 0:
        print("NvlinkTransport: not device found")
        return False

    for device_id in range(num_devices):
        # 获取设备
        device = cuda.Device(device_id)
        
        # 检查设备是否支持 Fabric Memory
        try:
            # CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED 在 Python API 中为 Device attribute 22
            fabric_supported = device.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY)
            if fabric_supported == 0:
                return False
        except cuda.Error as e:
            print(f"NvlinkTransport: cuDeviceGetAttribute failed: {e}")
            return False

    return True

# 测试函数
if support_fabric_mem():
    print("The system supports Fabric Memory")
else:
    print("The system does NOT support Fabric Memory")