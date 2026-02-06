from __future__ import annotations

from .base import FrameworkAdapter
from .sfl_adapter import SFLAdapter
from .gas_adapter import GASAdapter
from .multisfl_adapter import MultiSFLAdapter


def get_adapter(framework: str) -> FrameworkAdapter:
    name = framework.lower()
    if name == "sfl":
        return SFLAdapter()
    if name == "gas":
        return GASAdapter()
    if name == "multisfl":
        return MultiSFLAdapter()
    raise ValueError(f"Unsupported framework: {framework}")
