import pytest
from ai_sdk.openrouter import openrouter
from ai_sdk import generate_text
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import math
import random

MODELS = [
    "openai/gpt-4o",
    "mistralai/mistral-small-3.1-24b-instruct",
    "google/gemma-3-4b-it",
    "microsoft/phi-4-multimodal-instruct"
]

TOOL_MODELS = [
    "mistralai/mistral-small-3.1-24b-instruct",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4.5-preview"
]

# Load environment variables from .env file
load_dotenv()

def pytest_configure(config):
    """Verify environment before running tests"""
    if not os.getenv("OPENROUTER_API_KEY"):
        raise Exception("OPENROUTER_API_KEY environment variable is not set")

@pytest.mark.parametrize("model", MODELS)
def test_generate_text(model):
    """Test basic text generation for each model"""
    response = generate_text(model=openrouter(model), prompt="Hello, world!")
    
    assert response.text is not None and len(response.text) > 0
    assert response.usage.total_tokens > 0

@pytest.mark.parametrize("model", MODELS)
def test_generate_text_with_image(model):
    """Test basic text generation for each model"""
    response = generate_text(
        model=openrouter(model),
        messages=[{
            "role": "user",
            "content": [{
                "type": "image",
                "image": "https://fastly.picsum.photos/id/237/200/300.jpg?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U"
            }]
        }]
    )
    
    assert response.text is not None and len(response.text) > 0
    assert response.usage.total_tokens > 0

@pytest.mark.parametrize("model", TOOL_MODELS)
def test_generate_text_with_tool(model):
    """Test text generation with tool for each model"""
    class WeatherTool(BaseModel):
        location: str

    response = generate_text(
        model=openrouter(model),
        prompt="What is the weather in San Francisco?",
        max_steps=2,
        tools={
            "weather": {
                "description": "Get the weather in a location",
                "parameters": WeatherTool,
                "execute": lambda location: {
                    "location": location,
                    "temperature": 72 + math.floor(random.random() * 21) - 10,
                }
            }
        }
    )

    assert response.text is not None and len(response.text) > 0
    assert response.usage.total_tokens > 0
