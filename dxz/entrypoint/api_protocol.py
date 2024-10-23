from typing import Optional, List
from pydantic import BaseModel, Field

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    n: Optional[int] = 1
    max_tokens: Optional[int] = 16
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1

class CompletionResponseChoice(BaseModel):
    index: int
    text: str

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]