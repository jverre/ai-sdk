from .types import TextResult, Message, Usage, RequestMetadata, ResponseMetadata, AssistantMessage, Tool, ResponseMessage, TextPart, ToolCallPart, ToolResultPart, ToolMessage
from .utils import standardize_messages
from typing import List, Optional, Dict, Literal, Any, Callable
from .language_model import LanguageModel, LanguageModelCallOptions, LanguageModelProviderMetadata
from .tool_calls import execute_tool_calls
from .errors import AI_APICallError
import time

def _to_response_messages(
    text: Optional[str] = "",
    tools: Dict[str, Any] = None,
    tool_calls: List[ToolCallPart] = None,
    tool_results: List[ToolResultPart] = None,
    message_id: str = None,
    generate_message_id: Callable[[], str] = None,
) -> List[ResponseMessage]:
    """
    Converts the result of a generateText call to a list of response messages.
    """
    tool_calls = tool_calls or []
    tool_results = tool_results or []
    response_messages: List[ResponseMessage] = []

    # Create assistant message
    content = [
        TextPart(type="text", text=text),
        *tool_calls
    ]

    response_messages.append(
        AssistantMessage(
            role="assistant",
            content=content,
            id=message_id
        )
    )

    # Add tool results if any exist
    if tool_results:
        response_messages.append(
            ToolMessage(
                role="tool",
                content=tool_results,
                id=generate_message_id()
            )
        )

    return response_messages

def generate_text(
    model: LanguageModel,
    system: Optional[str] = None,
    prompt: Optional[str] = None,
    messages: Optional[List[Message]] = None,
    tools: Optional[Dict[str, Tool]] = None,
    tool_choice: Optional[Literal["auto", "none"]] = "auto",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
    seed: Optional[int] = None,
    max_retries: int = 3,
    headers: Optional[Dict[str, str]] = None,
    max_steps: int = 1,
    provider_options: Optional[LanguageModelProviderMetadata] = None,
) -> TextResult:
    if not isinstance(model, LanguageModel):
        raise ValueError("model must be a LanguageModel")
    
    messages = standardize_messages(system, prompt, messages)

    if tool_choice == "none":
        tools = None
    
    options = LanguageModelCallOptions(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop_sequences=stop_sequences,
        seed=seed,
        max_retries=max_retries,
        tools=tools,
        headers=headers,
        provider_metadata=provider_options
    )
    step = 0
    
    while True:
        retry_count = 0
        while retry_count < options.max_retries:
            try:
                res = model.do_generate(options)
                break
            except AI_APICallError as e:
                if not e.is_retryable:
                    raise e
                else:
                    retry_count += 1
                    time.sleep(1.0 * (2 ** retry_count))
                    continue
            except Exception as e:
                raise e
            
        step += 1

        if res.tool_calls:
            tool_results = execute_tool_calls(res.tool_calls, tools)
            next_step_type = "tool-result"
        else:
            tool_results = []
            next_step_type = "done"
        
        if step >= max_steps:
            break
        else:
            if next_step_type == "tool-result":
                options.messages.append(
                    AssistantMessage(
                        content="",
                        tool_calls=res.tool_calls
                    )
                )
                for tool_result in tool_results:
                    options.messages.append(
                        ToolMessage(
                            content=tool_result.result,
                            tool_call_id=tool_result.tool_call_id
                        )
                    )
            else:
                break
        
    final_text = res.text or ''

    response_messages = _to_response_messages(
        final_text,
        tools,
        res.tool_calls,
        tool_results,
        res.response.id,
        lambda: res.response.id
    )

    return TextResult(
        text=final_text,
        tool_calls=res.tool_calls or [],
        tool_results=tool_results,
        usage=Usage(
            prompt_tokens=res.usage.prompt_tokens,
            completion_tokens=res.usage.completion_tokens,
            total_tokens=res.usage.prompt_tokens + res.usage.completion_tokens
        ),
        request=RequestMetadata(body=res.request.body),
        response=ResponseMetadata(
            id=res.response.id,
            model=res.response.model_id,
            timestamp=res.response.timestamp,
            headers=res.response.headers,
            body=res.response.body,
            messages=response_messages
        ),
        warnings=res.warnings
    )
