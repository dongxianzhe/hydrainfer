from enum import IntEnum, auto

class ScenarioType(IntEnum):
    Relaxed = 0
    Strict = 1

    def __str__(self):
        return self.name

class ScenarioClassifier:
    def __init__(self):
        pass

    def classify(self, n_text_tokens: int, n_output_tokens: int) -> ScenarioType:
        if n_text_tokens < 100 and n_output_tokens < 100:
            return ScenarioType.Strict
        return ScenarioType.Relaxed