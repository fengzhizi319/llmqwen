"""
Routers module
"""
from .chat import router as chat_router
from .completions import router as completions_router
from .code import router as code_router
from .models import router as models_router
from .health import router as health_router

__all__ = [
    "chat_router",
    "completions_router",
    "code_router",
    "models_router",
    "health_router",
]
