"""HalluStream: flag the facts an open LLM is probably making up, from its hidden states."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hallustream.detector import Detector, Entity, Event, Report

__version__ = "0.1.0a0"
__all__ = ["Detector", "Entity", "Event", "Report", "__version__"]


def __getattr__(name: str):
    # Loaded on first use, so the research code can import hallustream.* without the library
    if name in ("Detector", "Entity", "Event", "Report"):
        from hallustream import detector

        return getattr(detector, name)
    raise AttributeError(name)
