from .types import TextResult, Message, Usage, RequestMetadata, ResponseMetadata
from .utils import standardize_messages
from typing import List, Optional, Dict
from .language_model import LanguageModel, LanguageModelCallOptions, LanguageModelProviderMetadata, LanguageModelCallResult
from .errors import AI_ObjectValidationError, AI_UnsupportedFunctionalityError, AI_APICallError
from .convert_response import convert_to_response_messages
import time
from pydantic import BaseModel
from .types import Tool, ObjectResult
import json
import opik

@opik.track
def _parse_responses(object_generation_mode: str, res: LanguageModelCallResult, schema: BaseModel) -> BaseModel:
    if object_generation_mode == "json" or object_generation_mode == "text":
        if res.text:
            try:
                object = json.loads(res.text)
            except Exception as e:
                raise AI_ObjectValidationError(
                    message=f"Failed to parse JSON object - text: {res.text}"
                )
            
            try:
                object = schema.model_validate(object)
            except Exception as e:
                raise AI_ObjectValidationError(
                    message=f"Failed to convert JSON object to Pydantic object - object: {object}"
                )
        else:
            raise AI_ObjectValidationError(
                message=f"No JSON object returned - text: {res.text}"
            )
    
    elif object_generation_mode == "tool_call":
        if len(res.tool_calls) > 1:
            raise AI_ObjectValidationError(
                message=f"Multiple tool calls returned, expected 1 - tool_calls: {res.tool_calls}"
            )
        elif len(res.tool_calls) == 0:
            raise AI_ObjectValidationError(
                message=f"No tool calls returned, expected 1 - tool_calls: {res.tool_calls}"
            )
        else:
            tool_call = res.tool_calls[0]
            if tool_call.args:
                try:
                    object = schema.model_validate(tool_call.args)
                except Exception as e:
                    raise AI_ObjectValidationError(
                        message=f"Failed to parse tool call result - args: {tool_call.args}"
                    )
            else:
                raise AI_ObjectValidationError(
                    message=f"No tool call result returned 0 args - tool_call: {tool_call}"
                )
    else:
        raise AI_UnsupportedFunctionalityError(
            functionality="This model does not support object generation"
        )

    return object

@opik.track
def _inject_json_schema(prompt: Optional[str], schema: BaseModel) -> str:
    DEFAULT_SCHEMA_PREFIX = 'JSON schema:'
    DEFAULT_SCHEMA_SUFFIX = "You MUST answer with a JSON object that matches the JSON schema above. Do not include any other text, only the JSON object and DO NOT return the data in markdown format."

    components = [
        prompt if prompt and len(prompt) > 0 else None,
        '' if prompt and len(prompt) > 0 else None,  # newline after prompt if it exists
        DEFAULT_SCHEMA_PREFIX,
        json.dumps(schema.model_json_schema()),
        DEFAULT_SCHEMA_SUFFIX
    ]
    
    # Filter out None values and join with newlines
    return '\n'.join(line for line in components if line is not None)

@opik.track
def generate_object(
    model: LanguageModel,
    schema: BaseModel,
    schema_name: Optional[str] = None,
    schema_description: Optional[str] = None,
    system: Optional[str] = None,
    prompt: Optional[str] = None,
    messages: Optional[List[Message]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    max_retries: int = 3,
    headers: Optional[Dict[str, str]] = None,
    provider_options: Optional[LanguageModelProviderMetadata] = None,
) -> TextResult:
    if not isinstance(model, LanguageModel):
        raise ValueError("model must be a LanguageModel")
    
    # if not model.supports_json_mode() and not model.supports_tool_calls():
    #     raise AI_UnsupportedFunctionalityError(
    #         functionality="This model does not support object generation"
    #     )

    options = LanguageModelCallOptions(
        messages=[], # Placeholder - This is updated below
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop_sequences=None,
        seed=seed,
        max_retries=max_retries,
        tools=None,
        headers=headers,
        provider_metadata=provider_options,
    )

    if model.supports_json_mode():
        options.response_format = schema
        object_generation_mode = "json"
    elif model.supports_tool_calls():
        schema_name = schema_name or "json_object"
        options.tools = {
            schema_name: Tool(
                description=schema_description or "Generate a JSON object",
                parameters=schema
            )
        }
        object_generation_mode = "tool_call"
    else:
        system = _inject_json_schema(system, schema)
        object_generation_mode = "text"

    messages = standardize_messages(system, prompt, messages)
    options.messages = messages
    
    while True:
        retry_count = 0
        while retry_count < options.max_retries:
            try:
                res = model.do_generate(options)
                object = _parse_responses(object_generation_mode, res, schema)
                break
            except AI_APICallError as e:
                if not e.is_retryable:
                    raise e
                else:
                    retry_count += 1
                    if retry_count >= options.max_retries:
                        raise e

                    time.sleep(1.0 * (2 ** retry_count))
                    continue
            except AI_ObjectValidationError as e:
                raise e
            except Exception as e:
                raise e
            
        break

    response_messages = convert_to_response_messages(
        res.text,
        options.tools,
        res.tool_calls,
        [],
        res.response.id,
        lambda: res.response.id
    )

    return ObjectResult(
        object=object,
        finish_reason=res.finish_reason,
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
        warnings=res.warnings,
        provider_metadata=res.provider_metadata
    )
