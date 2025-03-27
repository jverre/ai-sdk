from ..core.language_model import LanguageModel, LanguageModelCallOptions, LanguageModelCallResult, LanguageModelUsage, LanguageModelRequest, LanguageModelResponse, LanguageModelFinishReason
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel
from ..core.types import UnsupportedSettingWarning, Message, ToolCallPart
from ..core.errors import AI_UnsupportedFunctionalityError, AI_APICallError
import validators
import json
import datetime
import uuid

SUPPORTED_MODELS = [
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

SUPPORTED_IMAGE_MODELS = [
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

SUPPORTED_TOOL_MODELS = SUPPORTED_MODELS

class AnthropicChatSettings(BaseModel):
    sendReasoning: Optional[bool] = True

class AnthropicChatConfig(BaseModel):
    provider: str
    url: Any
    headers: Any
    fetch: Any

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class MessageBlock(BaseModel):
    type: str
    messages: List[Message]

class AnthropicChatModel(LanguageModel):
    def __init__(self, model_id: str, settings: AnthropicChatSettings, config: AnthropicChatConfig):
        if model_id not in SUPPORTED_MODELS:
            raise AI_UnsupportedFunctionalityError(
                functionality="Model",
                message=f"This model is not supported: {model_id}"
            )
        self.settings = settings
        self.config = config
        self.default_object_generation_mode = "tool"
        super().__init__(model_id, config.provider)

    def group_into_blocks(self, messages: List[Message]) -> List[MessageBlock]:
        """Group messages into blocks by role."""
        blocks = []
        current_block = None

        for message in messages:
            if message.role == MessageRole.SYSTEM:
                if not current_block or current_block.type != "system":
                    current_block = MessageBlock(type="system", messages=[])
                    blocks.append(current_block)
                current_block.messages.append(message)
                
            elif message.role == MessageRole.ASSISTANT:
                if not current_block or current_block.type != "assistant":
                    current_block = MessageBlock(type="assistant", messages=[])
                    blocks.append(current_block)
                current_block.messages.append(message)
                
            elif message.role == MessageRole.USER:
                if not current_block or current_block.type != "user":
                    current_block = MessageBlock(type="user", messages=[])
                    blocks.append(current_block)
                current_block.messages.append(message)
            elif message.role == MessageRole.TOOL:
                if not current_block or current_block.type != "tool":
                    current_block = MessageBlock(type="tool", messages=[])
                    blocks.append(current_block)
                current_block.messages.append(message)
            else:
                raise AI_UnsupportedFunctionalityError(
                    functionality=f"Unsupported message type: {message.role}",
                )

        return blocks

    def _convert_messages(self, messages: List[Message]) -> Dict[str, Any]:
        system = None
        blocks = self.group_into_blocks(messages)

        res = []
        for block in blocks:
            if block.type == "system":
                if system is not None:
                    raise AI_UnsupportedFunctionalityError(
                        functionality="Multiple system messages that are separated by user/assistant messages",
                    )
                elif len(res) > 0:
                    raise AI_UnsupportedFunctionalityError(
                        functionality="System messages must be the first messages",
                    )
                else:
                    system = [{"type": "text", "text": message.content} for message in block.messages]

            elif block.type == "assistant":
                content = []
                for message in block.messages:
                    if message.content != "":
                        content.append({
                            "type": "text",
                            "text": message.content
                        })
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            content.append({
                                "type": "tool_use",
                                "id": tool_call.tool_call_id,
                                "name": tool_call.tool_name,
                                "input": tool_call.args
                            })
                res.append({
                    "role": "assistant",
                    "content": content
                })
            elif block.type == "user":
                content = []
                for message in block.messages:
                    if isinstance(message.content, str):
                        content.append({
                            "type": "text",
                            "text": message.content
                        })
                    else:
                        for part in message.content:
                            if part.type == "text":
                                content.append({
                                    "type": "text",
                                    "text": part.text
                                })
                            elif part.type == "image":
                                if self.model_id not in SUPPORTED_IMAGE_MODELS:
                                    raise AI_UnsupportedFunctionalityError(
                                        functionality="Image input",
                                        message="This model does not support image input"
                                    )
                                if validators.url(part.image):
                                    content.append({
                                        "type": "image",
                                        "source": {
                                            "type": "url",
                                            "url": part.image
                                        }
                                    })
                                else:
                                    content.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": part.mime_type or "image/jpeg",
                                            "data": part.image
                                        }
                                    })
                res.append({
                    "role": "user",
                    "content": content
                })
            elif block.type == "tool":
                content = []
                for message in block.messages:
                    content.append({
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content
                    })
                res.append({
                    "role": "user",
                    "content": content
                })
            else:
                raise AI_UnsupportedFunctionalityError(
                    functionality=f"Unsupported message type: {block.type}",
                )

        return res

    def _get_args(self, options: LanguageModelCallOptions):
        warnings = []

        args = {}
        args["model"] = self.model_id
        args["messages"] = self._convert_messages(options.messages)
        args["max_tokens"] = options.max_tokens or 4096

        if options.temperature is not None:
            args["temperature"] = options.temperature
        if options.stop_sequences is not None:
            args["stop"] = options.stop_sequences
        if options.top_p is not None:
            args["top_p"] = options.top_p
        
        if options.top_k is not None:
            warnings.append(UnsupportedSettingWarning(
                setting="top_k"
            ))

        if options.presence_penalty is not None:
            warnings.append(UnsupportedSettingWarning(
                setting="presence_penalty"
            ))

        if options.frequency_penalty is not None:
            warnings.append(UnsupportedSettingWarning(
                setting="frequency_penalty"
            ))

        if options.seed is not None:
            warnings.append(UnsupportedSettingWarning(
                setting="seed"
            ))

        if options.tools is not None:
            args["tools"] = [{
                "name": tool_name,
                "description": tool.description,
                "input_schema": tool.parameters.model_json_schema()
            } for tool_name, tool in options.tools.items()]

        return args, warnings
        
    def _get_provider_metadata(self, response: Dict[str, Any]) -> Dict[str, Any]:
        provider_metadata = {"anthropic": {}}

        completion_token_details = response.get("usage", {}).get("completion_tokens_details")
        prompt_token_details = response.get("usage", {}).get("prompt_tokens_details")

        if completion_token_details:
            if completion_token_details.get("reasoning_tokens") is not None:
                provider_metadata["openai"]["reasoning_tokens"] = completion_token_details["reasoning_tokens"]
            
            if completion_token_details.get("accepted_prediction_tokens") is not None:
                provider_metadata["openai"]["accepted_prediction_tokens"] = completion_token_details["accepted_prediction_tokens"]
            
            if completion_token_details.get("rejected_prediction_tokens") is not None:
                provider_metadata["openai"]["rejected_prediction_tokens"] = completion_token_details["rejected_prediction_tokens"]

        if prompt_token_details and prompt_token_details.get("cached_tokens") is not None:
            provider_metadata["openai"]["cached_prompt_tokens"] = prompt_token_details["cached_tokens"]

        return provider_metadata

    def _is_retryable(self, response_code: int) -> bool:
        if response_code in [408, 409, 429] or response_code >= 500:
            return True
        
        return False

    def _convert_finish_reason(self, response: Dict[str, Any]) -> LanguageModelFinishReason:
        if response["stop_reason"] == "end_turn":
            return "stop"
        elif response["stop_reason"] == "stop_sequence":
            return "stop"
        elif response["stop_reason"] == "max_tokens":
            return "length"
        else:
            return "unknown"

    def _parse_tool_calls(self, result: Any) -> List[ToolCallPart]:
        """Parse tool calls from Anthropic response"""
        tool_calls = []
        
        for content in result.get("content", []):
            if content.get("type") == "tool_use":
                tool_calls.append(ToolCallPart(
                    tool_call_id=content["id"],
                    type="tool-call",
                    tool_name=content["name"],
                    args=content["input"]
                ))
        
        return tool_calls

    def do_generate(self, options: LanguageModelCallOptions) -> LanguageModelCallResult:
        args, warnings = self._get_args(options)
        
        with self.config.fetch() as client:
            url = self.config.url("/v1/messages")
            response = client.post(
                url = url,
                headers = self.config.headers(),
                json = args,
                timeout = 60
            )

            result = response.json()
            if response.status_code != 200:
                if self._is_retryable(response.status_code):
                    retry_count = 0
                    while retry_count < options.max_retries:
                        response = client.post(
                            url = url,
                            headers = self.config.headers(),
                            json = args,
                            timeout = 60
                        )

                        if response.status_code == 200:
                            break
                        else:
                            if not self._is_retryable(response.status_code):
                                raise AI_APICallError(
                                    url = url,
                                    request_body_values = args,
                                    status_code = response.status_code,
                                    response_headers = response.headers,
                                    response_body = result,
                                    is_retryable = self._is_retryable(response.status_code)
                                )
                            else:
                                retry_count += 1
                else:
                    raise AI_APICallError(
                        url = url,
                        request_body_values = args,
                        status_code = response.status_code,
                        response_headers = response.headers,
                        response_body = result,
                        is_retryable = self._is_retryable(response.status_code)
                    )
            
            return LanguageModelCallResult(
                text = result["content"][0]["text"] if result["content"][0]["type"] == "text" else "",
                tool_calls = self._parse_tool_calls(result),
                usage = LanguageModelUsage(
                    prompt_tokens = result["usage"]["input_tokens"],
                    completion_tokens = result["usage"]["output_tokens"]
                ),
                request = LanguageModelRequest(
                    body = json.dumps(args)
                ),
                response = LanguageModelResponse(
                    id = result["id"],
                    finish_reason = self._convert_finish_reason(result),
                    timestamp = datetime.datetime.now(),
                    headers = response.headers,
                    model_id = result["model"],
                    body = json.dumps(result)
                ),
                warnings = warnings,
                provider_metadata = self._get_provider_metadata(result)
            )