import pytest
from ai_sdk.anthropic import anthropic
from ai_sdk.anthropic.chat_model import SUPPORTED_MODELS
from ai_sdk import generate_object
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from ai_sdk.core.errors import AI_UnsupportedFunctionalityError
import math
import random
from typing import List

# Load environment variables from .env file
load_dotenv()

def pytest_configure(config):
    """Verify environment before running tests"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise Exception("ANTHROPIC_API_KEY environment variable is not set")

@pytest.mark.parametrize("model_id", SUPPORTED_MODELS)
def test_generate_object(model_id):
    """Test basic text generation for each model"""
    # Define the schema using Pydantic
    class Recipe(BaseModel):
        name: str
        ingredients: List[str]
        steps: List[str]

    class RecipeResponse(BaseModel):
        recipe: Recipe

    model = anthropic(model_id)
    if model.supports_tool_calls() or model.supports_json_mode():
        response = generate_object(
            model=model,
            schema=RecipeResponse,
            prompt="Generate a lasagna recipe.",
            max_tokens=10000
        )
        
        assert response.object is not None
        assert response.usage.total_tokens > 0
    else:
        with pytest.raises(AI_UnsupportedFunctionalityError):
            generate_object(
                model=model,
                schema=RecipeResponse,
                prompt="Generate a lasagna recipe.",
                max_tokens=10000
            )