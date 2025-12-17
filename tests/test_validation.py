"""Tests for input validation module."""

import pytest

from graph_layout import Group, Link, Node
from graph_layout.validation import (
    InvalidCanvasSizeError,
    InvalidGroupError,
    InvalidLinkError,
    ValidationError,
    validate_alpha,
    validate_canvas_size,
    validate_group_indices,
    validate_iterations,
    validate_link_indices,
)


class TestCanvasSizeValidation:
    """Tests for canvas size validation."""

    def test_valid_size(self):
        """Valid canvas size returns tuple."""
        w, h = validate_canvas_size([800, 600])
        assert w == 800.0
        assert h == 600.0

    def test_valid_size_floats(self):
        """Float values are accepted."""
        w, h = validate_canvas_size([800.5, 600.5])
        assert w == 800.5
        assert h == 600.5

    def test_negative_width_raises(self):
        """Negative width raises InvalidCanvasSizeError."""
        with pytest.raises(InvalidCanvasSizeError, match="width must be positive"):
            validate_canvas_size([-100, 600])

    def test_negative_height_raises(self):
        """Negative height raises InvalidCanvasSizeError."""
        with pytest.raises(InvalidCanvasSizeError, match="height must be positive"):
            validate_canvas_size([800, -100])

    def test_zero_width_raises(self):
        """Zero width raises InvalidCanvasSizeError."""
        with pytest.raises(InvalidCanvasSizeError, match="width must be positive"):
            validate_canvas_size([0, 600])

    def test_zero_height_raises(self):
        """Zero height raises InvalidCanvasSizeError."""
        with pytest.raises(InvalidCanvasSizeError, match="height must be positive"):
            validate_canvas_size([800, 0])

    def test_empty_size_raises(self):
        """Empty list raises InvalidCanvasSizeError."""
        with pytest.raises(InvalidCanvasSizeError, match="must have 2 elements"):
            validate_canvas_size([])

    def test_single_element_raises(self):
        """Single element raises InvalidCanvasSizeError."""
        with pytest.raises(InvalidCanvasSizeError, match="must have 2 elements"):
            validate_canvas_size([800])


class TestLinkValidation:
    """Tests for link index validation."""

    def test_valid_links(self):
        """Valid links return empty issues list."""
        links = [Link(0, 1), Link(1, 2)]
        issues = validate_link_indices(links, node_count=3)
        assert issues == []

    def test_valid_links_with_node_objects(self):
        """Links with Node objects validate correctly."""
        n0 = Node(index=0)
        n1 = Node(index=1)
        links = [Link(n0, n1)]
        issues = validate_link_indices(links, node_count=2)
        assert issues == []

    def test_out_of_bounds_source_strict(self):
        """Out of bounds source raises in strict mode."""
        links = [Link(10, 1)]
        with pytest.raises(InvalidLinkError, match="source index 10 out of bounds"):
            validate_link_indices(links, node_count=3, strict=True)

    def test_out_of_bounds_target_strict(self):
        """Out of bounds target raises in strict mode."""
        links = [Link(0, 99)]
        with pytest.raises(InvalidLinkError, match="target index 99 out of bounds"):
            validate_link_indices(links, node_count=3, strict=True)

    def test_negative_source_raises(self):
        """Negative source index raises."""
        links = [Link(-1, 1)]
        with pytest.raises(InvalidLinkError, match="source index -1 out of bounds"):
            validate_link_indices(links, node_count=3, strict=True)

    def test_non_strict_returns_issues(self):
        """Non-strict mode returns issues list without raising."""
        links = [Link(0, 1), Link(10, 20)]
        issues = validate_link_indices(links, node_count=3, strict=False)
        # Link 1 has both source (10) and target (20) out of bounds
        assert len(issues) == 2

    def test_empty_links_valid(self):
        """Empty links list is valid."""
        issues = validate_link_indices([], node_count=0)
        assert issues == []

    def test_boundary_indices_valid(self):
        """Boundary indices (0 and n-1) are valid."""
        links = [Link(0, 2)]
        issues = validate_link_indices(links, node_count=3)
        assert issues == []


class TestGroupValidation:
    """Tests for group index validation."""

    def test_valid_groups(self):
        """Valid groups return empty issues list."""
        groups = [Group(leaves=[0, 1])]
        issues = validate_group_indices(groups, node_count=3)
        assert issues == []

    def test_invalid_leaf_index_strict(self):
        """Invalid leaf index raises in strict mode."""
        groups = [Group(leaves=[0, 99])]
        with pytest.raises(InvalidGroupError, match="leaf index 99 out of bounds"):
            validate_group_indices(groups, node_count=3, strict=True)

    def test_non_strict_returns_issues(self):
        """Non-strict mode returns issues without raising."""
        groups = [Group(leaves=[0, 99])]
        issues = validate_group_indices(groups, node_count=3, strict=False)
        assert len(issues) == 1

    def test_empty_groups_valid(self):
        """Empty groups list is valid."""
        issues = validate_group_indices([], node_count=0)
        assert issues == []

    def test_group_without_leaves(self):
        """Group without leaves is valid."""
        groups = [Group()]
        issues = validate_group_indices(groups, node_count=3)
        assert issues == []


class TestIterationsValidation:
    """Tests for iterations validation."""

    def test_valid_iterations(self):
        """Valid iterations returns value."""
        assert validate_iterations(100) == 100

    def test_minimum_iterations(self):
        """Minimum iteration (1) is valid."""
        assert validate_iterations(1) == 1

    def test_zero_iterations_raises(self):
        """Zero iterations raises ValidationError."""
        with pytest.raises(ValidationError, match="must be >= 1"):
            validate_iterations(0)

    def test_negative_iterations_raises(self):
        """Negative iterations raises ValidationError."""
        with pytest.raises(ValidationError, match="must be >= 1"):
            validate_iterations(-10)


class TestAlphaValidation:
    """Tests for alpha validation."""

    def test_valid_alpha(self):
        """Valid alpha returns value."""
        assert validate_alpha(0.5) == 0.5

    def test_alpha_zero(self):
        """Alpha of 0 is valid."""
        assert validate_alpha(0.0) == 0.0

    def test_alpha_one(self):
        """Alpha of 1 is valid."""
        assert validate_alpha(1.0) == 1.0

    def test_alpha_negative_raises(self):
        """Negative alpha raises ValidationError."""
        with pytest.raises(ValidationError, match="must be in"):
            validate_alpha(-0.1)

    def test_alpha_greater_than_one_raises(self):
        """Alpha > 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="must be in"):
            validate_alpha(1.1)


class TestLinkConstructorValidation:
    """Tests for Link constructor validation."""

    def test_link_none_source_raises(self):
        """Link with None source raises ValueError."""
        with pytest.raises(ValueError, match="source cannot be None"):
            Link(None, 1)  # type: ignore

    def test_link_none_target_raises(self):
        """Link with None target raises ValueError."""
        with pytest.raises(ValueError, match="target cannot be None"):
            Link(0, None)  # type: ignore

    def test_link_valid_int_indices(self):
        """Link with valid int indices works."""
        link = Link(0, 1)
        assert link.source == 0
        assert link.target == 1

    def test_link_valid_node_objects(self):
        """Link with valid Node objects works."""
        n0 = Node(index=0)
        n1 = Node(index=1)
        link = Link(n0, n1)
        assert link.source is n0
        assert link.target is n1


class TestBaseLayoutValidation:
    """Tests for BaseLayout validation integration."""

    def test_size_validation_rejects_negative(self):
        """BaseLayout size property rejects negative dimensions."""
        from graph_layout import FruchtermanReingoldLayout

        with pytest.raises(InvalidCanvasSizeError):
            FruchtermanReingoldLayout(size=(-100, 600))

    def test_size_validation_rejects_zero(self):
        """BaseLayout size property rejects zero dimensions."""
        from graph_layout import CircularLayout

        with pytest.raises(InvalidCanvasSizeError):
            CircularLayout(size=(0, 600))

    def test_validate_method_catches_bad_links(self):
        """BaseLayout.validate() catches invalid link indices."""
        from graph_layout import FruchtermanReingoldLayout

        layout = FruchtermanReingoldLayout(
            nodes=[{}, {}],  # 2 nodes
            links=[{"source": 0, "target": 99}],  # Invalid target
        )

        with pytest.raises(InvalidLinkError):
            layout.validate()

    def test_validate_method_passes_valid_config(self):
        """BaseLayout.validate() passes valid configuration."""
        from graph_layout import FruchtermanReingoldLayout

        layout = FruchtermanReingoldLayout(
            nodes=[{}, {}, {}],
            links=[{"source": 0, "target": 1}, {"source": 1, "target": 2}],
        )

        # Should not raise
        result = layout.validate()
        assert result is layout  # Returns self for chaining
