import pytest
import socket
import subprocess
from hydrainfer.utils.socket_utils import find_interface_by_ip


def test_find_interface_by_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    interface = find_interface_by_ip(target_ip=ip_address)

    command = f"ip addr show | grep -B2 '{ip_address}'"  + """| head -n 1 | awk '{sub(/:$/, "", $2); print $2}'"""
    print(f'command: {command}')
    interface_ref = subprocess.run(command, shell=True, text=True, capture_output=True).stdout.strip()

    print(f'ip_address {ip_address} interface {interface}')
    assert interface == interface_ref


if __name__ == '__main__':
    pytest.main([__file__, '-s'])