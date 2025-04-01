from typing import List, Optional, Dict, Any, Callable
from .types import AssistantMessage, ResponseMessage, TextPart, ToolCallPart, ToolResultPart, ToolMessage
import opik

@opik.track
def convert_to_response_messages(
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
