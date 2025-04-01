from pydantic import BaseModel
from typing import Dict, Callable, Any, Optional
from ..core.utils import load_api_key
from .chat_model import OpenRouterChatModel, OpenRouterChatSettings, OpenRouterChatConfig
import httpx
from urllib.parse import urljoin

class OpenRouterProviderSettings(BaseModel):
    name: str = "openrouter"
    base_url: Optional[str] = "https://openrouter.ai/api"
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    fetch: Optional[Callable[[str], Any]] = httpx.Client

class OpenRouterProvider:
    def __init__(self, settings: OpenRouterProviderSettings):
        self.settings = settings

        self.chat = self.create_chat_model
    
    def _get_headers(self) -> Dict[str, str]:
        openrouter_headers = {
            "Authorization": f"Bearer {load_api_key(self.settings.api_key, 'OPENROUTER_API_KEY', 'OpenRouter')}",
            "HTTP-Referer": "https://github.com/jverre/ai-sdk",
            "X-Title": "AI SDK - Python"
        }

        if self.settings.headers is not None:
            openrouter_headers.update(self.settings.headers)
        
        return openrouter_headers
    
    def _join_url(self, path: str) -> str:
        # Ensure base_url ends with a slash if it has a path
        base = self.settings.base_url
        
        if not base.endswith('/'):
            base = base + '/'
        
        if path.startswith('/'):
            path = path[1:]

        return base + path
    
    def create_chat_model(self, model_id: str, settings: OpenRouterChatSettings) -> OpenRouterChatModel:
        return OpenRouterChatModel(
            model_id=model_id,
            settings=settings,
            config=OpenRouterChatConfig(
                provider=f"{self.settings.name}.chat",
                url=self._join_url,
                headers=self._get_headers,
                fetch=self.settings.fetch
            )
        )
    
    def __call__(self, model_id: str, settings: OpenRouterChatSettings) -> OpenRouterChatModel:
        return self.chat(model_id, settings)

def openrouter(model_id: str, settings: OpenRouterChatSettings = OpenRouterChatSettings()) -> OpenRouterChatModel:
    return OpenRouterProvider(
        settings=OpenRouterProviderSettings()
    ).chat(model_id, settings)

def create_openrouter_provider(settings: OpenRouterProviderSettings) -> OpenRouterProvider:
    return OpenRouterProvider(settings)