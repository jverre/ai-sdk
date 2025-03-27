from typing import Optional, List, Dict, Literal, Any
from .types import Message, Warning, ToolCallPart, Tool
from pydantic import BaseModel
import datetime

class LanguageModelCallSettings(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None

LanguageModelProviderMetadata = Dict[str, Dict[str, Any]]

class LanguageModelCallOptions(LanguageModelCallSettings):
    tools: Optional[Dict[str, Tool]] = None
    messages: List[Message]
    headers: Optional[Dict[str, str]] = None
    provider_metadata: Optional[LanguageModelProviderMetadata] = None
    max_retries: int = 3

LanguageModelFinishReason = Literal["stop", "length", "content_filter", "error", "other", "unknown"]

class LanguageModelUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

class LanguageModelRequest(BaseModel):
    body: Optional[str] = None

class LanguageModelResponse(BaseModel):
    id: Optional[str] = None
    timestamp: Optional[datetime.datetime] = None
    headers: Optional[Dict[str, str]] = None
    model_id: Optional[str] = None
    body: Optional[str] = None


class LanguageModelCallResult(BaseModel):
    text: Optional[str] = None
    tool_calls: Optional[List[ToolCallPart]] = None
    finish_reason: Optional[LanguageModelFinishReason] = None
    usage: LanguageModelUsage
    request: Optional[LanguageModelRequest] = None
    response: Optional[LanguageModelResponse] = None
    warnings: Optional[List[Warning]] = None
    provider_metadata: Optional[LanguageModelProviderMetadata] = None


class LanguageModel:
    required_attributes = {
        'model_id',
        'provider',
        'default_object_generation_mode'
    }

    def __init__(self, model_id: str, provider: str):
        self.model_id = model_id
        self.provider = provider
        self._validate_attributes()

    def _validate_attributes(self) -> None:
        missing = [attr for attr in self.required_attributes 
                  if not hasattr(self, attr)]
        if missing:
            raise AttributeError(
                f"Missing required attributes: {', '.join(missing)}"
            )

    def do_generate(self, options: LanguageModelCallOptions) -> LanguageModelCallResult:
        pass
    