from pydantic import BaseModel
from utils import PIISchema
import json
from typing import Optional

def valid_json(text: str) -> tuple[bool, Optional[str]]:
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3]
    try:
        json.loads(text)
    except Exception as e:
        return False, f"Failed to parse JSON, response is not a valid JSON object: error: {e}"
    
    return True, None
