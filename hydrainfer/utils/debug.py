from functools import wraps
import inspect
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

def save_checkpoint(save_path, only_save_once=True):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if only_save_once and not hasattr(self, 'only_save_once'):
                self.only_save_once=True
                sig = inspect.signature(func)
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()

                input = {name: arg.detach().clone() if torch.is_tensor(arg) else arg for name, arg in bound_args.arguments.items()}
                
                output = func(self, *args, **kwargs)
                
                # output_data = output.detach().clone()
                
                torch.save({
                    'input': input,
                    'output': output,
                }, save_path)
                
                print(f"forward input and output have been saved to {save_path}.")
            else:
                output = func(self, *args, **kwargs)
            return output
        return wrapper
    return decorator