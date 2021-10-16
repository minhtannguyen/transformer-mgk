"""This module implements a basic event system that allows the transformer
internal components to make available any tensor with minimal overhead."""

from .event import Event, AttentionEvent, QKVEvent
from .event_dispatcher import EventDispatcher
