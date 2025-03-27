from pydantic import BaseModel
from typing import Dict, Callable, Any, Optional
from ..core.utils import load_api_key
from .chat_model import AnthropicChatModel, AnthropicChatSettings, AnthropicChatConfig
import httpx
from urllib.parse import urljoin

class AnthropicProviderSettings(BaseModel):
    name: str = "anthropic"
    base_url: Optional[str] = "https://api.anthropic.com/v1"
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    fetch: Optional[Callable[[str], Any]] = httpx.Client

class AnthropicProvider:
    def __init__(self, settings: AnthropicProviderSettings):
        self.settings = settings

        self.chat = self.create_chat_model
    
    def _get_headers(self) -> Dict[str, str]:
        anthropic_headers = {
            "x-api-key": f"{load_api_key(self.settings.api_key, 'ANTHROPIC_API_KEY', 'Anthropic')}",
            "anthropic-version": "2023-06-01"
        }

        if self.settings.headers is not None:
            anthropic_headers.update(self.settings.headers)
        
        return anthropic_headers
    
    def create_chat_model(self, model_id: str, settings: AnthropicChatSettings) -> AnthropicChatModel:
        return AnthropicChatModel(
            model_id=model_id,
            settings=settings,
            config=AnthropicChatConfig(
                provider=f"{self.settings.name}.chat",
                url=lambda path: urljoin(self.settings.base_url, path),
                headers=self._get_headers,
                fetch=self.settings.fetch
            )
        )
    
    def __call__(self, model_id: str, settings: AnthropicChatSettings) -> AnthropicChatModel:
        return self.chat(model_id, settings)

def anthropic(model_id: str, settings: AnthropicChatSettings = AnthropicChatSettings()) -> AnthropicChatModel:
    return AnthropicProvider(
        settings=AnthropicProviderSettings()
    ).chat(model_id, settings)

def create_anthropic_provider(settings: AnthropicProviderSettings) -> AnthropicProvider:
    return AnthropicProvider(settings)