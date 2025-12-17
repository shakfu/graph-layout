"""
Input validation utilities for graph layout algorithms.

Provides centralized validation functions for nodes, links, canvas size,
and other layout parameters. Raises descriptive exceptions on invalid input.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


class ValidationError(ValueError):
    """Base exception for layout validation errors."""

    pass


class InvalidCanvasSizeError(ValidationError):
    """Raised when canvas dimensions are invalid."""

    pass


class InvalidNodeError(ValidationError):
    """Raised when a node is malformed."""

    pass


class InvalidLinkError(ValidationError):
    """Raised when a link references invalid nodes."""

    pass


class InvalidGroupError(ValidationError):
    """Raised when a group references invalid nodes/groups."""

    pass


def validate_canvas_size(size: Sequence[float]) -> tuple[float, float]:
    """
    Validate canvas size dimensions.

    Args:
        size: [width, height] sequence

    Returns:
        Validated (width, height) tuple

    Raises:
        InvalidCanvasSizeError: If dimensions are invalid
    """
    if len(size) < 2:
        raise InvalidCanvasSizeError(
            f"Canvas size must have 2 elements [width, height], got {len(size)}"
        )

    width, height = float(size[0]), float(size[1])

    if width <= 0:
        raise InvalidCanvasSizeError(f"Canvas width must be positive, got {width}")
    if height <= 0:
        raise InvalidCanvasSizeError(f"Canvas height must be positive, got {height}")

    return width, height


def validate_link_indices(
    links: Sequence[Any],
    node_count: int,
    strict: bool = True,
) -> list[tuple[int, str]]:
    """
    Validate that all link source/target indices are within bounds.

    Args:
        links: Sequence of Link objects or dicts with source/target
        node_count: Number of nodes in the graph
        strict: If True, raises on invalid. If False, returns list of issues.

    Returns:
        List of (link_index, issue_description) tuples

    Raises:
        InvalidLinkError: If strict=True and invalid links found
    """
    issues: list[tuple[int, str]] = []

    for i, link in enumerate(links):
        src = _get_index(link, "source")
        tgt = _get_index(link, "target")

        if src is None:
            issues.append((i, f"Link {i}: source is None"))
        elif src < 0 or src >= node_count:
            issues.append((i, f"Link {i}: source index {src} out of bounds [0, {node_count})"))

        if tgt is None:
            issues.append((i, f"Link {i}: target is None"))
        elif tgt < 0 or tgt >= node_count:
            issues.append((i, f"Link {i}: target index {tgt} out of bounds [0, {node_count})"))

    if strict and issues:
        msg = "Invalid link indices:\n" + "\n".join(issue[1] for issue in issues)
        raise InvalidLinkError(msg)

    return issues


def validate_group_indices(
    groups: Sequence[Any],
    node_count: int,
    strict: bool = True,
) -> list[tuple[int, str]]:
    """
    Validate that group leaf/group indices are within bounds.

    Args:
        groups: Sequence of Group objects
        node_count: Number of nodes
        strict: If True, raises on invalid

    Returns:
        List of (group_index, issue_description) tuples

    Raises:
        InvalidGroupError: If strict=True and invalid groups found
    """
    issues: list[tuple[int, str]] = []
    group_count = len(groups)

    for gi, group in enumerate(groups):
        leaves = getattr(group, "leaves", None)
        subgroups = getattr(group, "groups", None)

        if leaves:
            for leaf in leaves:
                idx = _get_index_simple(leaf)
                if idx is not None and (idx < 0 or idx >= node_count):
                    issues.append(
                        (
                            gi,
                            f"Group {gi}: leaf index {idx} out of bounds [0, {node_count})",
                        )
                    )

        if subgroups:
            for subgroup in subgroups:
                idx = _get_index_simple(subgroup)
                if idx is not None and (idx < 0 or idx >= group_count):
                    issues.append(
                        (
                            gi,
                            f"Group {gi}: subgroup index {idx} out of bounds [0, {group_count})",
                        )
                    )

    if strict and issues:
        msg = "Invalid group indices:\n" + "\n".join(issue[1] for issue in issues)
        raise InvalidGroupError(msg)

    return issues


def validate_iterations(iterations: int) -> int:
    """
    Validate iteration count is positive.

    Args:
        iterations: Number of iterations

    Returns:
        Validated iteration count

    Raises:
        ValidationError: If iterations < 1
    """
    if iterations < 1:
        raise ValidationError(f"iterations must be >= 1, got {iterations}")
    return iterations


def validate_alpha(alpha: float) -> float:
    """
    Validate alpha is in valid range.

    Args:
        alpha: Alpha value

    Returns:
        Validated alpha value

    Raises:
        ValidationError: If alpha not in [0, 1]
    """
    if alpha < 0 or alpha > 1:
        raise ValidationError(f"alpha must be in [0, 1], got {alpha}")
    return alpha


def _get_index(obj: Any, attr: str) -> Optional[int]:
    """Extract index from int, Node, or object with index attribute."""
    if hasattr(obj, attr):
        val = getattr(obj, attr, None)
    elif isinstance(obj, dict):
        val = obj.get(attr)
    else:
        val = None

    if val is None:
        return None
    if isinstance(val, int):
        return val
    if hasattr(val, "index"):
        return int(val.index)
    return None


def _get_index_simple(obj: Any) -> Optional[int]:
    """Extract index from int or object with index attribute."""
    if isinstance(obj, int):
        return obj
    if hasattr(obj, "index"):
        return int(obj.index)
    return None


__all__ = [
    "ValidationError",
    "InvalidCanvasSizeError",
    "InvalidNodeError",
    "InvalidLinkError",
    "InvalidGroupError",
    "validate_canvas_size",
    "validate_link_indices",
    "validate_group_indices",
    "validate_iterations",
    "validate_alpha",
]
