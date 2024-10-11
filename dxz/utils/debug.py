import torch

def print_once(scope:str="default", message: str=""):
    if not hasattr(print_once, 'scope_printed'):
        print_once.scope_printed: dict[str, bool] = {}
    if scope not in print_once.scope_printed:
        print_once.scope_printed[scope]=False
    if not print_once.scope_printed[scope]:
        print_once.scope_printed[scope]=True
        print(f"{scope}: {message}")

def probe(key: str, value: any):
    if not hasattr(probe, 'data'):
        probe.data = {}
    if key not in probe.data:
        probe.data[key] = value
    if not torch.allclose(probe.data[key], value, rtol=1e-3, atol=1e-5):
        print(f'probe {key} not equal !!! ')


if __name__ == '__main__':
    x = torch.zeros(size=(10, 128))
    y = torch.zeros(size=(10, 128))
    probe('h', x)
    probe('h', y)