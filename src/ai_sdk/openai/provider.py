from pydantic import BaseModel
from typing import Dict, Callable, Any, Optional
from ..core.utils import load_api_key
from .chat_model import OpenAIChatModel, OpenAIChatSettings, OpenAIChatConfig
import httpx
from urllib.parse import urljoin, urlparse

class OpenAIProviderSettings(BaseModel):
    name: str = "openai"
    base_url: Optional[str] = "https://api.openai.com"
    api_key: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    fetch: Optional[Callable[[str], Any]] = httpx.Client

class OpenAIProvider:
    def __init__(self, settings: OpenAIProviderSettings):
        self.settings = settings

        self.chat = self.create_chat_model
    
    def _get_headers(self) -> Dict[str, str]:
        openai_headers = {
            "Authorization": f"Bearer {load_api_key(self.settings.api_key, 'OPENAI_API_KEY', 'OpenAI')}",
        }

        if self.settings.organization is not None:
            openai_headers['OpenAI-Organization'] = self.settings.organization
        
        if self.settings.project is not None:
            openai_headers['OpenAI-Project'] = self.settings.project
        
        if self.settings.headers is not None:
            openai_headers.update(self.settings.headers)
        
        return openai_headers
    
    def _join_url(self, path: str) -> str:
        # Ensure base_url ends with a slash if it has a path
        base = self.settings.base_url
        
        if not base.endswith('/'):
            base = base + '/'
        
        if path.startswith('/'):
            path = path[1:]

        return base + path
    
    def create_chat_model(self, model_id: str, settings: OpenAIChatSettings) -> OpenAIChatModel:
        return OpenAIChatModel(
            model_id=model_id,
            settings=settings,
            config=OpenAIChatConfig(
                provider=f"{self.settings.name}.chat",
                url=self._join_url,
                headers=self._get_headers,
                fetch=self.settings.fetch
            )
        )
    
    def __call__(self, model_id: str, settings: OpenAIChatSettings) -> OpenAIChatModel:
        return self.chat(model_id, settings)

def openai(model_id: str, settings: OpenAIChatSettings = OpenAIChatSettings()) -> OpenAIChatModel:
    return OpenAIProvider(
        settings=OpenAIProviderSettings()
    ).chat(model_id, settings)

def create_openai_provider(settings: OpenAIProviderSettings) -> OpenAIProvider:
    return OpenAIProvider(settings)