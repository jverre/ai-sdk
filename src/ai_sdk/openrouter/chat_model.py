from ..core.language_model import LanguageModel, LanguageModelCallOptions, LanguageModelCallResult, LanguageModelUsage, LanguageModelRequest, LanguageModelResponse
from typing import Optional, Dict, Union, Any, List
from pydantic import BaseModel
from ..core.types import Message, ToolCallPart, FinishReason
from ..core.errors import AI_APICallError
import json
import datetime
import validators
import opik

class OpenRouterChatSettings(BaseModel):
    logit_bias: Optional[Dict[float, float]] = None
    log_probs: Optional[Union[int, bool]] = None
    parallel_tool_calls: Optional[bool] = None
    structured_outputs: Optional[bool] = None
    user: Optional[str] = None

class OpenRouterChatConfig(BaseModel):
    provider: str
    url: Any
    headers: Any
    fetch: Any

class OpenRouterChatModel(LanguageModel):
    def __init__(self, model_id: str, settings: OpenRouterChatSettings, config: OpenRouterChatConfig):
        self.default_object_generation_mode = "text"
        self.settings = settings
        self.config = config
        
        super().__init__(model_id, config.provider)
        

    def _convert_finish_reason(self, finish_reason: str) -> FinishReason:
        if finish_reason == "tool_calls":
            return "tool-calls"
        elif finish_reason == "content_filter":
            return "content-filter"
        else:
            return finish_reason
    
    @opik.track
    def _get_args(self, options: LanguageModelCallOptions):
        warnings = []

        args = {}
        args["model"] = self.model_id
        args["messages"] = self._convert_messages(options.messages)
        if options.max_tokens is not None:
            args["max_tokens"] = options.max_tokens
        if options.temperature is not None:
            args["temperature"] = options.temperature
        if options.stop_sequences is not None:
            args["stop"] = options.stop_sequences
        if options.top_p is not None:
            args["top_p"] = options.top_p
        if options.top_k is not None:
            args["top_k"] = options.top_k
        if options.presence_penalty is not None:
            args["presence_penalty"] = options.presence_penalty
        if options.frequency_penalty is not None:
            args["frequency_penalty"] = options.frequency_penalty
        
        if options.tools is not None:
            args["tools"] = [{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool.description,
                    "parameters": tool.parameters.model_json_schema()
                }
            } for tool_name, tool in options.tools.items()]
        
        if options.seed is not None:
            args["seed"] = options.seed

        return args, warnings
        
    def _is_retryable(self, response_code: int) -> bool:
        if response_code in [408, 409, 429] or response_code >= 500:
            return True
        
        return False

    def supports_json_mode(self) -> bool:
        return True

    def supports_tool_calls(self) -> bool:
        return True

    def _convert_tool_calls_to_openrouter_format(self, tool_calls: list[ToolCallPart]) -> list[Dict[str, Any]]:
        """
        Converts internal ToolCallPart format to OpenRouter's tool_calls format.
        
        Args:
            tool_calls: List of ToolCallPart objects
            
        Returns:
            List of tool calls in OpenRouter's format
        """
        openrouter_tool_calls = []
        
        for tool_call in tool_calls:
            openrouter_tool_calls.append({
                "id": tool_call.tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": json.dumps(tool_call.args)
                }
            })
        
        return openrouter_tool_calls

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        res = []

        for message in messages:
            if message.role == "system":
                res.append({
                    "role": "system",
                    "content":  message.content
                })
            elif message.role == "developer":
                res.append({
                    "role": "developer",
                    "content": message.content
                })
            elif message.role == "assistant":
                res.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": self._convert_tool_calls_to_openrouter_format(message.tool_calls or [])
                })
            elif message.role == "user":
                if isinstance(message.content, str):
                    res.append({
                        "role": "user",
                        "content": message.content
                    })
                else:
                    content = []
                    for part in message.content:
                        if part.type == "text":
                            content.append({
                                "type": "text",
                                "text": part.text
                            })
                        elif part.type == "image":
                            if validators.url(part.image):
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": part.image
                                    }
                                })
                            else:
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{part.mime_type or 'image/jpeg'};base64,{part.image}"
                                    }
                                })
                            
                    res.append({
                        "role": "user",
                        "content": content
                    })
            elif message.role == "assistant":
                res.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": self._convert_tool_calls_to_openai_format(message.tool_calls or [])
                })
            elif message.role == "tool":
                res.append({
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id
                })
        return res

    @opik.track
    def _parse_tool_calls(self, result: Any) -> List[ToolCallPart]:
        tool_calls = []

        for choice in result["choices"]:
            if choice["finish_reason"] == "tool_calls":
                for tool_call in choice["message"]["tool_calls"]:
                    # Parse the JSON string into a Python dict
                    args_dict = json.loads(tool_call["function"]["arguments"])
                    
                    tool_calls.append(ToolCallPart(
                        tool_call_id=tool_call["id"],
                        type="tool-call",
                        tool_name=tool_call["function"]["name"],
                        args=args_dict  # Now passing a dictionary instead of a string
                    ))
        return tool_calls

    @opik.track(type="llm")
    def do_generate(self, options: LanguageModelCallOptions) -> LanguageModelCallResult:
        args, warnings = self._get_args(options)

        with self.config.fetch() as client:
            print(self.config.url("/v1/chat/completions"))
            response = client.post(
                url = self.config.url("/v1/chat/completions"),
                headers = self.config.headers(),
                json = args,
                timeout = 60
            )

            result = response.json()
            if response.status_code != 200:
                raise AI_APICallError(
                    url = self.config.url("/v1/chat/completions"),
                    request_body_values = args,
                    status_code = response.status_code,
                    response_headers = response.headers,
                    response_body = result,
                    is_retryable = self._is_retryable(response.status_code)
                )
            
            if result.get("error", None) is not None:
                raise AI_APICallError(
                    url = self.config.url("/v1/chat/completions"),
                    request_body_values = args,
                    status_code = response.status_code,
                    response_headers = response.headers,
                    response_body = result,
                    is_retryable = self._is_retryable(response.status_code)
                )
            
            return LanguageModelCallResult(
                text = result["choices"][0]["message"]["content"],
                finish_reason = self._convert_finish_reason(result["choices"][0]["finish_reason"]),
                tool_calls = self._parse_tool_calls(result),
                usage = LanguageModelUsage(
                    prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
                ),
                request = LanguageModelRequest(
                    body = json.dumps(args)
                ),
                response = LanguageModelResponse(
                    id = result["id"],
                    timestamp = datetime.datetime.fromtimestamp(result["created"]),
                    headers = response.headers,
                    model_id = result["model"],
                    body = json.dumps(result)
                ),
                warnings = warnings
            )