"""Tests for VPSC constraint solver."""

from graph_layout.cola.vpsc import (
    Block,
    Blocks,
    Constraint,
    Solver,
    Variable,
    remove_overlap_in_one_dimension,
)


def round_to(v: float, precision: int = 4) -> float:
    """Round to specified precision."""
    m = 10**precision
    return round(v * m) / m


def get_positions(variables: list[Variable], precision: int = 4) -> list[float]:
    """Get rounded positions from variables."""
    return [round_to(v.position(), precision) for v in variables]


class TestVariable:
    """Test Variable class."""

    def test_create_variable(self):
        """Test variable creation."""
        v = Variable(5.0)
        assert v.desired_position == 5.0
        assert v.weight == 1.0
        assert v.scale == 1.0

    def test_variable_with_weight_scale(self):
        """Test variable with custom weight and scale."""
        v = Variable(10.0, weight=2.0, scale=3.0)
        assert v.desired_position == 10.0
        assert v.weight == 2.0
        assert v.scale == 3.0


class TestConstraint:
    """Test Constraint class."""

    def test_create_constraint(self):
        """Test constraint creation."""
        v1 = Variable(0.0)
        v2 = Variable(5.0)
        c = Constraint(v1, v2, 3.0)

        assert c.left is v1
        assert c.right is v2
        assert c.gap == 3.0
        assert not c.equality
        assert not c.active


class TestSolver:
    """Test VPSC Solver with comprehensive test cases."""

    def test_no_splits(self):
        """Test case: no splits required."""
        variables = [Variable(2), Variable(9), Variable(9), Variable(9), Variable(2)]
        constraints = [
            Constraint(variables[0], variables[4], 3),
            Constraint(variables[0], variables[1], 3),
            Constraint(variables[1], variables[2], 3),
            Constraint(variables[2], variables[4], 3),
            Constraint(variables[3], variables[4], 3),
        ]

        solver = Solver(variables, constraints)
        solver.solve()

        expected = [1.4, 4.4, 7.4, 7.4, 10.4]
        assert get_positions(variables) == expected

    def test_simple_scale(self):
        """Test case: simple scale."""
        variables = [
            Variable(0, weight=1, scale=2),
            Variable(0, weight=1, scale=1),
        ]
        constraints = [Constraint(variables[0], variables[1], 2)]

        solver = Solver(variables, constraints)
        solver.solve()

        expected = [-0.8, 0.4]
        assert get_positions(variables) == expected

    def test_simple_scale_2(self):
        """Test case: simple scale with 3 variables."""
        variables = [
            Variable(1, weight=1, scale=3),
            Variable(1, weight=1, scale=2),
            Variable(1, weight=1, scale=4),
        ]
        constraints = [
            Constraint(variables[0], variables[1], 2),
            Constraint(variables[1], variables[2], 2),
        ]

        solver = Solver(variables, constraints)
        solver.solve()

        expected = [0.2623, 1.3934, 1.1967]
        assert get_positions(variables) == expected

    def test_non_trivial_merging(self):
        """Test case: non-trivial merging."""
        variables = [Variable(x) for x in [4, 6, 9, 2, 5]]
        constraints = [
            Constraint(variables[0], variables[2], 3),
            Constraint(variables[0], variables[3], 3),
            Constraint(variables[1], variables[4], 3),
            Constraint(variables[2], variables[4], 3),
            Constraint(variables[2], variables[3], 3),
            Constraint(variables[3], variables[4], 3),
        ]

        solver = Solver(variables, constraints)
        solver.solve()

        expected = [0.5, 6, 3.5, 6.5, 9.5]
        assert get_positions(variables) == expected

    def test_case_5(self):
        """Test case 5."""
        variables = [Variable(x) for x in [5, 6, 7, 4, 3]]
        constraints = [
            Constraint(variables[0], variables[4], 3),
            Constraint(variables[1], variables[2], 3),
            Constraint(variables[2], variables[3], 3),
            Constraint(variables[2], variables[4], 3),
            Constraint(variables[3], variables[4], 3),
        ]

        solver = Solver(variables, constraints)
        solver.solve()

        expected = [5, 0.5, 3.5, 6.5, 9.5]
        assert get_positions(variables) == expected

    def test_split_block_to_activate_different_constraint(self):
        """Test case: split block to activate different constraint."""
        variables = [Variable(x) for x in [7, 1, 6, 0, 2]]
        constraints = [
            Constraint(variables[0], variables[3], 3),
            Constraint(variables[0], variables[1], 3),
            Constraint(variables[1], variables[4], 3),
            Constraint(variables[2], variables[4], 3),
            Constraint(variables[2], variables[3], 3),
            Constraint(variables[3], variables[4], 3),
        ]

        solver = Solver(variables, constraints)
        solver.solve()

        expected = [0.8, 3.8, 0.8, 3.8, 6.8]
        assert get_positions(variables) == expected

    def test_non_trivial_split(self):
        """Test case: non-trivial split."""
        variables = [Variable(x) for x in [0, 9, 1, 9, 5, 1, 2, 1, 6, 3]]
        constraints = [
            Constraint(variables[0], variables[3], 3),
            Constraint(variables[1], variables[8], 3),
            Constraint(variables[1], variables[6], 3),
            Constraint(variables[2], variables[6], 3),
            Constraint(variables[3], variables[5], 3),
            Constraint(variables[3], variables[6], 3),
            Constraint(variables[3], variables[7], 3),
            Constraint(variables[4], variables[8], 3),
            Constraint(variables[4], variables[7], 3),
            Constraint(variables[5], variables[8], 3),
            Constraint(variables[5], variables[7], 3),
            Constraint(variables[5], variables[8], 3),
            Constraint(variables[6], variables[9], 3),
            Constraint(variables[7], variables[8], 3),
            Constraint(variables[7], variables[9], 3),
            Constraint(variables[8], variables[9], 3),
        ]

        solver = Solver(variables, constraints)
        solver.solve()

        expected = [-3.714, 4.0, 1.0, -0.714, 2.286, 2.286, 7.0, 5.286, 8.286, 11.286]
        assert get_positions(variables, precision=3) == expected

    def test_case_7(self):
        """Test case 7."""
        variables = [Variable(x) for x in [7, 0, 3, 1, 4]]
        constraints = [
            Constraint(variables[0], variables[3], 3),
            Constraint(variables[0], variables[2], 3),
            Constraint(variables[1], variables[4], 3),
            Constraint(variables[1], variables[4], 3),
            Constraint(variables[2], variables[3], 3),
            Constraint(variables[3], variables[4], 3),
        ]

        solver = Solver(variables, constraints)
        solver.solve()

        expected = [-0.75, 0, 2.25, 5.25, 8.25]
        assert get_positions(variables) == expected

    def test_case_8(self):
        """Test case 8."""
        variables = [Variable(x) for x in [4, 2, 3, 1, 8]]
        constraints = [
            Constraint(variables[0], variables[4], 3),
            Constraint(variables[0], variables[2], 3),
            Constraint(variables[1], variables[3], 3),
            Constraint(variables[2], variables[3], 3),
            Constraint(variables[2], variables[4], 3),
            Constraint(variables[3], variables[4], 3),
        ]

        solver = Solver(variables, constraints)
        solver.solve()

        expected = [-0.5, 2, 2.5, 5.5, 8.5]
        assert get_positions(variables) == expected

    def test_case_9(self):
        """Test case 9."""
        variables = [Variable(x) for x in [3, 4, 0, 5, 6]]
        constraints = [
            Constraint(variables[0], variables[1], 3),
            Constraint(variables[0], variables[2], 3),
            Constraint(variables[1], variables[2], 3),
            Constraint(variables[1], variables[4], 3),
            Constraint(variables[2], variables[3], 3),
            Constraint(variables[2], variables[3], 3),
            Constraint(variables[3], variables[4], 3),
            Constraint(variables[3], variables[4], 3),
        ]

        solver = Solver(variables, constraints)
        solver.solve()

        expected = [-2.4, 0.6, 3.6, 6.6, 9.6]
        assert get_positions(variables) == expected

    def test_set_desired_positions(self):
        """Test updating desired positions."""
        variables = [Variable(0), Variable(5)]
        constraints = [Constraint(variables[0], variables[1], 3)]

        solver = Solver(variables, constraints)
        solver.solve()

        # Update desired positions
        solver.set_desired_positions([10, 15])
        solver.solve()

        # Variables should move toward new desired positions
        assert variables[0].desired_position == 10
        assert variables[1].desired_position == 15
        assert variables[1].position() - variables[0].position() >= 3

    def test_equality_constraint(self):
        """Test equality constraint."""
        variables = [Variable(0), Variable(10)]
        constraints = [Constraint(variables[0], variables[1], 5, equality=True)]

        solver = Solver(variables, constraints)
        solver.solve()

        # With equality, gap should be exactly 5
        gap = variables[1].position() - variables[0].position()
        assert abs(gap - 5.0) < 1e-6

    def test_unsatisfiable_constraint(self):
        """Test handling of unsatisfiable constraints (cycles)."""
        variables = [Variable(0), Variable(5), Variable(10)]

        # Create a cycle: v0 < v1 < v2 < v0
        constraints = [
            Constraint(variables[0], variables[1], 3),
            Constraint(variables[1], variables[2], 3),
            Constraint(variables[2], variables[0], 3),  # Creates cycle
        ]

        solver = Solver(variables, constraints)
        cost = solver.solve()

        # One constraint should be marked unsatisfiable
        unsatisfiable_count = sum(1 for c in constraints if c.unsatisfiable)
        assert unsatisfiable_count > 0


class TestRemoveOverlap:
    """Test overlap removal function."""

    def test_simple_overlap_removal(self):
        """Test removing overlap from simple span configuration."""
        spans = [
            {"size": 4, "desiredCenter": 5},
            {"size": 4, "desiredCenter": 6},
            {"size": 4, "desiredCenter": 7},
        ]

        result = remove_overlap_in_one_dimension(spans)

        # Check that spans don't overlap
        centers = result["newCenters"]
        for i in range(len(centers) - 1):
            gap = centers[i + 1] - centers[i]
            min_gap = (spans[i]["size"] + spans[i + 1]["size"]) / 2
            assert gap >= min_gap - 1e-6

    def test_overlap_removal_with_bounds(self):
        """Test overlap removal with lower and upper bounds."""
        spans = [
            {"size": 2, "desiredCenter": 5},
            {"size": 2, "desiredCenter": 6},
        ]

        result = remove_overlap_in_one_dimension(spans, lower_bound=0, upper_bound=10)

        centers = result["newCenters"]

        # Check bounds are respected
        assert centers[0] >= result["lowerBound"] + spans[0]["size"] / 2
        assert centers[-1] <= result["upperBound"] - spans[-1]["size"] / 2

        # Check no overlap
        gap = centers[1] - centers[0]
        min_gap = (spans[0]["size"] + spans[1]["size"]) / 2
        assert gap >= min_gap - 1e-6

    def test_overlap_removal_returns_bounds(self):
        """Test that function returns appropriate bounds."""
        spans = [
            {"size": 4, "desiredCenter": 5},
            {"size": 4, "desiredCenter": 6},
        ]

        result = remove_overlap_in_one_dimension(spans)

        assert "newCenters" in result
        assert "lowerBound" in result
        assert "upperBound" in result
        assert len(result["newCenters"]) == len(spans)


class TestBlocks:
    """Test Blocks class."""

    def test_blocks_creation(self):
        """Test creating blocks from variables."""
        variables = [Variable(i) for i in range(3)]
        blocks = Blocks(variables)

        # Each variable should initially be in its own block
        assert len(blocks._list) == 3

    def test_blocks_cost(self):
        """Test cost calculation across blocks."""
        variables = [Variable(0), Variable(10)]
        blocks = Blocks(variables)

        cost = blocks.cost()
        # Cost should be sum of squared deviations
        # Each variable starts at desired position (0 deviation)
        assert cost == 0

    def test_block_merge(self):
        """Test merging blocks."""
        v1 = Variable(0)
        v2 = Variable(10)
        blocks = Blocks([v1, v2])

        # Create and activate a constraint
        c = Constraint(v1, v2, 5)
        c.active = True
        v1.block.vars[0].offset = 0
        v2.block.vars[0].offset = 5

        initial_count = len(blocks._list)
        blocks.merge(c)

        # Should have one less block after merge
        assert len(blocks._list) == initial_count - 1


class TestBlock:
    """Test Block class."""

    def test_block_creation(self):
        """Test creating a block."""
        v = Variable(5)
        block = Block(v)

        assert len(block.vars) == 1
        assert block.vars[0] is v
        assert v.block is block

    def test_block_cost(self):
        """Test block cost calculation."""
        v = Variable(5, weight=2)
        block = Block(v)

        # Variable at desired position
        block.posn = 5 / v.scale
        assert abs(block.cost()) < 1e-6

    def test_is_active_directed_path(self):
        """Test checking for active directed paths."""
        v1 = Variable(0)
        v2 = Variable(5)
        v3 = Variable(10)

        block = Block(v1)

        # No path from v1 to v1... well, it's the same variable
        assert block.is_active_directed_path_between(v1, v1)
