from typing import List, Optional, Literal, Union, Dict, Any, Type, Callable
from pydantic import BaseModel
import datetime

FinishReason = Literal["stop", "length", "content-filter", "tool-calls", "error", "other", "unknown"]

class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str

class ToolResultPart(BaseModel):
    type: Literal["tool-result"] = "tool-result"
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: Optional[bool] = None

class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str

class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ToolCallPart(BaseModel):
    type: Literal["tool-call"] = "tool-call"
    tool_call_id: str
    tool_name: str
    args: Dict[str, Any]

class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    image: str
    mime_type: Optional[str] = None

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: Union[str, List[Union[TextPart, ImagePart]]]

class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Union[str, List[Union[TextPart, ToolCallPart]]]
    tool_calls: Optional[List[ToolCallPart]] = None

Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]
ResponseMessage = Union[AssistantMessage, ToolMessage]

class RequestMetadata(BaseModel):
    body: str

class ResponseMetadata(BaseModel):
    id: str
    model: str
    timestamp: datetime.datetime
    headers: Optional[Dict[str, str]] = None
    body: Optional[Any] = None
    messages: List[ResponseMessage]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class UnsupportedSettingWarning(BaseModel):
    type: Literal["unsupported-setting"] = "unsupported-setting"
    setting: str

class OtherWarning(BaseModel):
    type: Literal["other"] = "other"
    message: str

Warning = Union[UnsupportedSettingWarning, OtherWarning]

class StepResult(BaseModel):
    step_type: Literal["initial", "continue", "tool-result"]
    text: str
    tool_calls: List[ToolCallPart]
    tool_results: List[ToolResultPart]
    finish_reason: FinishReason
    usage: Usage
    request: Optional[RequestMetadata] = None
    response: Optional[ResponseMetadata] = None
    warnings: Optional[List[Warning]] = None
    provider_metadata: Optional[Dict[str, Dict[str, Any]]] = None

class TextResult(BaseModel):
    text: str
    usage: Usage
    tool_calls: List[ToolCallPart] = []
    tool_results: List[ToolResultPart] = []
    request: Optional[RequestMetadata] = None
    response: Optional[ResponseMetadata] = None
    warnings: Optional[List[Warning]] = None

class Tool(BaseModel):
    description: Optional[str] = None
    parameters: Type[BaseModel]
    execute: Callable[..., Any]
