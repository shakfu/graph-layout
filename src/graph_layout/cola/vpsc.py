"""
VPSC (Variable Placement with Separation Constraints) solver.

This module implements a constraint solver for maintaining separation constraints
between variables while minimizing a quadratic cost function.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


class PositionStats:
    """Maintains statistics for computing optimal block position."""

    def __init__(self, scale: float):
        self.scale = scale
        self.AB: float = 0.0
        self.AD: float = 0.0
        self.A2: float = 0.0

    def add_variable(self, v: Variable) -> None:
        """Add a variable to the statistics."""
        ai = self.scale / v.scale
        bi = v.offset / v.scale
        wi = v.weight
        self.AB += wi * ai * bi
        self.AD += wi * ai * v.desired_position
        self.A2 += wi * ai * ai

    def get_posn(self) -> float:
        """Compute optimal position from statistics."""
        return (self.AD - self.AB) / self.A2


class Constraint:
    """Separation constraint between two variables."""

    def __init__(self, left: Variable, right: Variable, gap: float, equality: bool = False):
        self.left = left
        self.right = right
        self.gap = gap
        self.equality = equality
        self.lm: float = 0.0  # Lagrangian multiplier
        self.active: bool = False
        self.unsatisfiable: bool = False

    def slack(self) -> float:
        """
        Compute slack (how much constraint is violated or satisfied).

        Returns:
            Positive if constraint is satisfied, negative if violated
        """
        if self.unsatisfiable:
            return float("inf")
        return (
            self.right.scale * self.right.position()
            - self.gap
            - self.left.scale * self.left.position()
        )


class Variable:
    """A variable with desired position, weight, and scale."""

    def __init__(self, desired_position: float, weight: float = 1.0, scale: float = 1.0):
        self.desired_position = desired_position
        self.weight = weight
        self.scale = scale
        self.offset: float = 0.0
        self.block: Optional[Block] = None
        self.c_in: list[Constraint] = []  # Constraints where this is on the right
        self.c_out: list[Constraint] = []  # Constraints where this is on the left

    def dfdv(self) -> float:
        """Derivative of cost function with respect to this variable."""
        return 2.0 * self.weight * (self.position() - self.desired_position)

    def position(self) -> float:
        """Compute current position of this variable."""
        assert self.block is not None, "Variable must be in a block to get position"
        return (self.block.ps.scale * self.block.posn + self.offset) / self.scale

    def visit_neighbours(
        self, prev: Optional[Variable], f: Callable[[Constraint, Variable], None]
    ) -> None:
        """
        Visit neighbors connected by active constraints within the same block.

        Args:
            prev: Previous variable (to avoid backtracking)
            f: Function to call for each (constraint, next_variable) pair
        """

        def visitor(c: Constraint, next_var: Variable) -> None:
            if c.active and prev is not next_var:
                f(c, next_var)

        for c in self.c_out:
            visitor(c, c.right)
        for c in self.c_in:
            visitor(c, c.left)


class Block:
    """
    A block of variables connected by active constraints.

    Variables in a block move together to satisfy active constraints.
    """

    def __init__(self, v: Variable):
        self.vars: list[Variable] = []
        self.posn: float = 0.0
        self.ps: PositionStats = PositionStats(v.scale)
        self.block_ind: int = 0
        v.offset = 0.0
        self._add_variable(v)

    def _add_variable(self, v: Variable) -> None:
        """Add a variable to this block."""
        v.block = self
        self.vars.append(v)
        self.ps.add_variable(v)
        self.posn = self.ps.get_posn()

    def update_weighted_position(self) -> None:
        """Recompute block position to minimize cost."""
        self.ps.AB = self.ps.AD = self.ps.A2 = 0.0
        for v in self.vars:
            self.ps.add_variable(v)
        self.posn = self.ps.get_posn()

    def _compute_lm(
        self, v: Variable, u: Optional[Variable], post_action: Callable[[Constraint], None]
    ) -> float:
        """
        Recursively compute Lagrangian multipliers for active constraints.

        Args:
            v: Current variable
            u: Previous variable (to avoid backtracking)
            post_action: Function to call after processing each constraint

        Returns:
            Derivative contribution
        """
        dfdv = v.dfdv()

        def process_neighbour(c: Constraint, next_var: Variable) -> None:
            nonlocal dfdv
            _dfdv = self._compute_lm(next_var, v, post_action)
            if next_var is c.right:
                dfdv += _dfdv * c.left.scale
                c.lm = _dfdv
            else:
                dfdv += _dfdv * c.right.scale
                c.lm = -_dfdv
            post_action(c)

        v.visit_neighbours(u, process_neighbour)
        return dfdv / v.scale

    def _populate_split_block(self, v: Variable, prev: Optional[Variable]) -> None:
        """
        Populate a new block after splitting.

        Args:
            v: Starting variable
            prev: Previous variable (to avoid backtracking)
        """

        def process_neighbour(c: Constraint, next_var: Variable) -> None:
            next_var.offset = v.offset + (c.gap if next_var is c.right else -c.gap)
            self._add_variable(next_var)
            self._populate_split_block(next_var, v)

        v.visit_neighbours(prev, process_neighbour)

    def traverse(
        self,
        visit: Callable[[Constraint], Any],
        acc: list[Any],
        v: Optional[Variable] = None,
        prev: Optional[Variable] = None,
    ) -> None:
        """
        Traverse active constraint tree, applying visit to each constraint.

        Args:
            visit: Function to apply to each constraint
            acc: Accumulator for results
            v: Current variable (defaults to first in block)
            prev: Previous variable
        """
        if v is None:
            v = self.vars[0]

        def process_neighbour(c: Constraint, next_var: Variable) -> None:
            acc.append(visit(c))
            self.traverse(visit, acc, next_var, v)

        v.visit_neighbours(prev, process_neighbour)

    def find_min_lm(self) -> Optional[Constraint]:
        """
        Find the active constraint with minimum Lagrangian multiplier.

        Returns:
            Constraint with minimum LM, or None if no split candidate found
        """
        m: Optional[Constraint] = None

        def check_constraint(c: Constraint) -> None:
            nonlocal m
            if not c.equality and (m is None or c.lm < m.lm):
                m = c

        self._compute_lm(self.vars[0], None, check_constraint)
        return m

    def _find_min_lm_between(self, lv: Variable, rv: Variable) -> Optional[Constraint]:
        """Find minimum LM constraint between two variables."""
        self._compute_lm(lv, None, lambda c: None)
        m: Optional[Constraint] = None

        def check_on_path(c: Constraint, next_var: Variable) -> None:
            nonlocal m
            if not c.equality and c.right is next_var and (m is None or c.lm < m.lm):
                m = c

        self._find_path(lv, None, rv, check_on_path)
        return m

    def _find_path(
        self,
        v: Variable,
        prev: Optional[Variable],
        to: Variable,
        visit: Callable[[Constraint, Variable], None],
    ) -> bool:
        """
        Find path from v to 'to' variable, calling visit for each edge.

        Returns:
            True if path found
        """
        end_found = False

        def check_neighbour(c: Constraint, next_var: Variable) -> None:
            nonlocal end_found
            if not end_found and (next_var is to or self._find_path(next_var, v, to, visit)):
                end_found = True
                visit(c, next_var)

        v.visit_neighbours(prev, check_neighbour)
        return end_found

    def is_active_directed_path_between(self, u: Variable, v: Variable) -> bool:
        """
        Check if there's a directed path from u to v through active constraints.

        Returns:
            True if directed path exists
        """
        if u is v:
            return True
        for c in reversed(u.c_out):
            if c.active and self.is_active_directed_path_between(c.right, v):
                return True
        return False

    @staticmethod
    def split(c: Constraint) -> list[Block]:
        """
        Split a block by deactivating the specified constraint.

        Args:
            c: Constraint to split on

        Returns:
            List of two new blocks [left_block, right_block]
        """
        c.active = False
        return [Block._create_split_block(c.left), Block._create_split_block(c.right)]

    @staticmethod
    def _create_split_block(start_var: Variable) -> Block:
        """Create a new block starting from the given variable."""
        b = Block(start_var)
        b._populate_split_block(start_var, None)
        return b

    def split_between(self, vl: Variable, vr: Variable) -> Optional[dict[str, Any]]:
        """
        Find a split point between two variables.

        Returns:
            Dictionary with 'constraint', 'lb', and 'rb' keys, or None if no split point
        """
        c = self._find_min_lm_between(vl, vr)
        if c is not None:
            bs = Block.split(c)
            return {"constraint": c, "lb": bs[0], "rb": bs[1]}
        return None

    def merge_across(self, b: Block, c: Constraint, dist: float) -> None:
        """
        Merge another block into this block across a constraint.

        Args:
            b: Block to merge
            c: Constraint being activated
            dist: Distance offset
        """
        c.active = True
        for v in b.vars:
            v.offset += dist
            self._add_variable(v)
        self.posn = self.ps.get_posn()

    def cost(self) -> float:
        """Compute the cost (squared deviation from desired positions)."""
        total = 0.0
        for v in self.vars:
            d = v.position() - v.desired_position
            total += d * d * v.weight
        return total


class Blocks:
    """Collection of blocks for VPSC solver."""

    def __init__(self, vs: list[Variable]):
        self.vs = vs
        n = len(vs)
        self._list: list[Block] = []
        for i in reversed(range(n)):
            b = Block(vs[i])
            self._list.insert(0, b)
            b.block_ind = i

    def cost(self) -> float:
        """Total cost across all blocks."""
        return sum(b.cost() for b in self._list)

    def insert(self, b: Block) -> None:
        """Insert a new block."""
        b.block_ind = len(self._list)
        self._list.append(b)

    def remove(self, b: Block) -> None:
        """Remove a block from the collection."""
        last_idx = len(self._list) - 1
        swap_block = self._list[last_idx]
        self._list.pop()
        if b is not swap_block:
            self._list[b.block_ind] = swap_block
            swap_block.block_ind = b.block_ind

    def merge(self, c: Constraint) -> None:
        """
        Merge blocks on either side of a constraint.

        Merges the smaller block into the larger one.
        """
        left_block = c.left.block
        right_block = c.right.block
        assert left_block is not None and right_block is not None
        dist = c.right.offset - c.left.offset - c.gap

        if len(left_block.vars) < len(right_block.vars):
            right_block.merge_across(left_block, c, dist)
            self.remove(left_block)
        else:
            left_block.merge_across(right_block, c, -dist)
            self.remove(right_block)

    def for_each(self, f: Callable[[Block, int], None]) -> None:
        """Apply function to each block."""
        for i, b in enumerate(self._list):
            f(b, i)

    def update_block_positions(self) -> None:
        """Update positions of all blocks."""
        for b in self._list:
            b.update_weighted_position()

    def split(self, inactive: list[Constraint]) -> None:
        """Split blocks across constraints with negative Lagrangian multipliers."""
        self.update_block_positions()
        # Need to iterate over a copy since we modify the list
        blocks_to_process = list(self._list)
        for b in blocks_to_process:
            v = b.find_min_lm()
            if v is not None and v.lm < Solver.LAGRANGIAN_TOLERANCE:
                old_block = v.left.block
                assert old_block is not None
                for nb in Block.split(v):
                    self.insert(nb)
                self.remove(old_block)
                inactive.append(v)


class Solver:
    """
    VPSC (Variable Placement with Separation Constraints) solver.

    Solves for variable positions that satisfy separation constraints while
    minimizing squared deviation from desired positions.
    """

    LAGRANGIAN_TOLERANCE = -1e-4
    ZERO_UPPERBOUND = -1e-10

    def __init__(self, vs: list[Variable], cs: list[Constraint]):
        self.vs = vs
        self.cs = cs
        self.bs: Optional[Blocks] = None

        # Initialize constraint lists for each variable
        for v in vs:
            v.c_in = []
            v.c_out = []

        # Build constraint graph
        for c in cs:
            c.left.c_out.append(c)
            c.right.c_in.append(c)

        # All constraints start inactive
        self.inactive = [c for c in cs]
        for c in cs:
            c.active = False

    def cost(self) -> float:
        """Get current cost."""
        assert self.bs is not None
        return self.bs.cost()

    def set_starting_positions(self, ps: list[float]) -> None:
        """Set starting positions without changing desired positions."""
        self.inactive = list(self.cs)
        for c in self.cs:
            c.active = False
        self.bs = Blocks(self.vs)
        self.bs.for_each(lambda b, i: setattr(b, "posn", ps[i]))

    def set_desired_positions(self, ps: list[float]) -> None:
        """Update desired positions for variables."""
        for v, p in zip(self.vs, ps):
            v.desired_position = p

    def _most_violated(self) -> Optional[Constraint]:
        """
        Find the most violated inactive constraint.

        Returns:
            Most violated constraint, or None if all are satisfied
        """
        min_slack = float("inf")
        v: Optional[Constraint] = None
        delete_point = len(self.inactive)

        for i, c in enumerate(self.inactive):
            if c.unsatisfiable:
                continue

            slack = c.slack()
            if c.equality or slack < min_slack:
                min_slack = slack
                v = c
                delete_point = i
                if c.equality:
                    break

        # Remove from inactive list if it should become active
        if delete_point < len(self.inactive) and v is not None:
            if (min_slack < Solver.ZERO_UPPERBOUND and not v.active) or v.equality:
                self.inactive[delete_point] = self.inactive[-1]
                self.inactive.pop()

        return v

    def satisfy(self) -> None:
        """Satisfy constraints by building/updating block structure."""
        if self.bs is None:
            self.bs = Blocks(self.vs)

        self.bs.split(self.inactive)
        v = self._most_violated()

        while v is not None and (
            v.equality or (v.slack() < Solver.ZERO_UPPERBOUND and not v.active)
        ):
            lb = v.left.block
            rb = v.right.block
            assert lb is not None and rb is not None

            if lb is not rb:
                # Blocks are different, merge them
                self.bs.merge(v)
            else:
                # Constraint is within a block
                if lb.is_active_directed_path_between(v.right, v.left):
                    # Cycle detected - constraint is unsatisfiable
                    v.unsatisfiable = True
                    v = self._most_violated()
                    continue

                # Split the block
                split = lb.split_between(v.left, v.right)
                if split is not None:
                    self.bs.insert(split["lb"])
                    self.bs.insert(split["rb"])
                    self.bs.remove(lb)
                    self.inactive.append(split["constraint"])
                else:
                    # Couldn't find split point
                    v.unsatisfiable = True
                    v = self._most_violated()
                    continue

                # Check if constraint was indirectly satisfied by split
                if v.slack() >= 0:
                    self.inactive.append(v)
                else:
                    self.bs.merge(v)

            v = self._most_violated()

    def solve(self) -> float:
        """
        Solve the constraint problem.

        Returns:
            Final cost
        """
        self.satisfy()
        assert self.bs is not None
        last_cost = float("inf")
        cost = self.bs.cost()

        while abs(last_cost - cost) > 0.0001:
            self.satisfy()
            last_cost = cost
            cost = self.bs.cost()

        return cost


def remove_overlap_in_one_dimension(
    spans: list[dict[str, float]],
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> dict[str, Any]:
    """
    Remove overlap between spans while keeping centers close to desired positions.

    Args:
        spans: List of dicts with 'size' and 'desiredCenter' keys
        lower_bound: Optional lower bound
        upper_bound: Optional upper bound

    Returns:
        Dictionary with 'newCenters', 'lowerBound', and 'upperBound' keys
    """
    vs = [Variable(s["desiredCenter"]) for s in spans]
    cs: list[Constraint] = []
    n = len(spans)

    # Create separation constraints between adjacent spans
    for i in range(n - 1):
        left = spans[i]
        right = spans[i + 1]
        gap = (left["size"] + right["size"]) / 2
        cs.append(Constraint(vs[i], vs[i + 1], gap))

    # Handle bounds
    left_most = vs[0]
    right_most = vs[n - 1]
    left_most_size = spans[0]["size"] / 2
    right_most_size = spans[n - 1]["size"] / 2
    v_lower: Optional[Variable] = None
    v_upper: Optional[Variable] = None

    if lower_bound is not None:
        v_lower = Variable(lower_bound, left_most.weight * 1000)
        vs.append(v_lower)
        cs.append(Constraint(v_lower, left_most, left_most_size))

    if upper_bound is not None:
        v_upper = Variable(upper_bound, right_most.weight * 1000)
        vs.append(v_upper)
        cs.append(Constraint(right_most, v_upper, right_most_size))

    # Solve
    solver = Solver(vs, cs)
    solver.solve()

    # Extract results
    new_centers = [v.position() for v in vs[:n]]
    result_lower = v_lower.position() if v_lower else left_most.position() - left_most_size
    result_upper = v_upper.position() if v_upper else right_most.position() + right_most_size

    return {
        "newCenters": new_centers,
        "lowerBound": result_lower,
        "upperBound": result_upper,
    }
