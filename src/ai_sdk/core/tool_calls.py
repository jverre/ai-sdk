from typing import List, Dict
from .types import ToolCallPart, ToolResultPart, Tool
from .errors import AI_ToolExecutionError
import json

def execute_tool_calls(
    tool_calls: List[ToolCallPart],
    tools: Dict[str, Tool]
) -> List[ToolResultPart]:
    results = []

    try:
        for tool_call in tool_calls:
            tool = tools[tool_call.tool_name]
            result = tool["execute"](**tool_call.args)
            results.append(
                ToolResultPart(
                    tool_call_id=tool_call.tool_call_id,
                    tool_name=tool_call.tool_name,
                    result=json.dumps(result)
                )
            )
    except Exception as e:
        raise AI_ToolExecutionError(
            tool_name=tool_call.tool_name,
            tool_args=tool_call.args,
            tool_call_id=tool_call.tool_call_id,
        )
    
    return results