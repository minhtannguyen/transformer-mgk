"""Allow for the dynamic registration of new attention implementations.

This module provides a Registry implementation that other modules can use to
register attention implementations for the builders.
"""

from .registry import \
    AttentionRegistry, \
    RecurrentAttentionRegistry, \
    RecurrentCrossAttentionRegistry
from .spec import Spec, Choice, Optional, Int, Float, Bool, Callable, \
    EventDispatcherInstance
