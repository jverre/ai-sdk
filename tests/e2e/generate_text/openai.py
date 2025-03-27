import pytest
from ai_sdk.openai import openai
from ai_sdk.openai.chat_model import SUPPORTED_MODELS, SUPPORTED_IMAGE_MODELS, SUPPORTED_TOOL_MODELS
from ai_sdk import generate_text
from ai_sdk.core.errors import AI_UnsupportedFunctionalityError
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import math
import random


# Load environment variables from .env file
load_dotenv()

def pytest_configure(config):
    """Verify environment before running tests"""
    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("OPENAI_API_KEY environment variable is not set")

@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_generate_text(model):
    """Test basic text generation for each model"""
    response = generate_text(model=openai(model), prompt="Hello, world!")
    
    assert response.text is not None and len(response.text) > 0
    assert response.usage.total_tokens > 0

@pytest.mark.parametrize("model", SUPPORTED_IMAGE_MODELS)
def test_generate_text_with_image(model):
    """Test basic text generation for each model"""
    response = generate_text(
        model=openai(model),
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


@pytest.mark.parametrize("model", set(SUPPORTED_MODELS) - set(SUPPORTED_IMAGE_MODELS))
def test_generate_text_with_image(model):
    """Test text generation with image for each model"""
    with pytest.raises(AI_UnsupportedFunctionalityError, match="This model does not support image input"):
        generate_text(
            model=openai(model),
            messages=[{
                "role": "user",
                "content": [{
                    "type": "image",
                    "image": "https://fastly.picsum.photos/id/237/200/300.jpg?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U"
                }]
            }]
        )

@pytest.mark.parametrize("model", SUPPORTED_TOOL_MODELS)
def test_generate_text_with_tool(model):
    """Test text generation with tool for each model"""
    class WeatherTool(BaseModel):
        location: str

    response = generate_text(
        model=openai(model),
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
