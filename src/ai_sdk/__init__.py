from .core.generate_text import generate_text
from .core.generate_object import generate_object
from .core.utils import is_opik_configured

is_opik_configured()

__all__ = ["generate_text", "generate_object"]

__version__ = "0.1.11"