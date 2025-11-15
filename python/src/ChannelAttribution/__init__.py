# src/ChannelAttribution/__init__.py

from . import _core as _core

# Esponi tutte le funzioni Cython del core
for name in dir(_core):
    if not name.startswith("_"):
        globals()[name] = getattr(_core, name)

# Esponi gli helper Python
from .install_pro import install_pro

__all__ = [name for name in globals() if not name.startswith("_")]
