"""Warning helpers.

Layouts warn about structural problems with the caller's graph -- a tree layout
handed a graph that is not a tree, say. Those warnings are only useful if they
are attributed to the line that called ``run()``, so that ``warnings`` filters
keyed on the caller's module match and the traceback points somewhere the caller
can act on.

Getting there with a literal ``stacklevel`` means counting library frames by
hand, and the counts differ per call path (they ranged from 2 to 4 here) and go
stale whenever an intermediate helper is added or removed. ``warn_at_caller``
counts the frames at runtime instead.
"""

from __future__ import annotations

import os
import sys
import warnings
from types import FrameType
from typing import Optional, Type

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def warn_at_caller(
    message: str,
    category: Optional[Type[Warning]] = None,
    extra_depth: int = 0,
) -> None:
    """Emit a warning attributed to the nearest frame outside this package.

    Args:
        message: Warning text.
        category: Warning class, as for ``warnings.warn``.
        extra_depth: Additional frames to skip, for callers that wrap this.

    Falls back to the immediate caller when every frame belongs to the package,
    which happens when the work runs on a worker thread (see
    ``base.run_deep_recursive``) and the calling frame is simply not on this
    stack.
    """
    frame: Optional[FrameType] = sys._getframe(1)
    depth = 1
    while frame is not None:
        filename = os.path.abspath(frame.f_code.co_filename)
        if os.path.dirname(filename) != _PACKAGE_DIR and not filename.startswith(
            _PACKAGE_DIR + os.sep
        ):
            break
        frame = frame.f_back
        depth += 1
    else:  # pragma: no cover - only when the stack is entirely internal
        depth = 1

    warnings.warn(message, category, stacklevel=depth + 1 + extra_depth)


__all__ = ["warn_at_caller"]
