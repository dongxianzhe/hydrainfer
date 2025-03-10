from enum import Enum, auto
from dxz.request import Request

class ScenarioType(Enum):
    Relaxed = auto()
    Strict = auto()

    def __str__(self):
        return self.name

class ScenarioClassifier:
    def __init__(self):
        pass

    def classify(self, n_prompt_tokens_without_image: int, max_tokens: int) -> ScenarioType:
        if n_prompt_tokens_without_image < 100 and max_tokens < 100:
            return ScenarioType.Strict
        return ScenarioType.Relaxed