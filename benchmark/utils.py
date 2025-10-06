import json
import dacite
from typing import TypeVar, Type

T = TypeVar("T")
def load_json(data_class: Type[T], data_path: str) -> T:
    with open(data_path) as file:
        data = json.load(file)
    config = dacite.from_dict(data_class=data_class, data=data)
    return config

