import shortuuid
import time
from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class ChatCompletionMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str
    image: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    max_tokens: Optional[int] = 16
    stream: Optional[bool] = False


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 16
    stream: Optional[bool] = False

class CompletionResponseChoice(BaseModel):
    index: int
    text: str

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]

class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str

class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]