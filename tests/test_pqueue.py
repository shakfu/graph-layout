"""Tests for priority queue (pairing heap) implementation."""

from graph_layout.cola.pqueue import PairingHeap, PriorityQueue


class TestPairingHeap:
    """Test cases for PairingHeap class."""

    def test_create_empty_heap(self):
        """Test creating an empty heap."""
        heap = PairingHeap(None)
        assert heap.empty()
        assert heap.count() == 0
        assert heap.min() is None

    def test_create_heap_with_element(self):
        """Test creating a heap with an element."""
        heap = PairingHeap(5)
        assert not heap.empty()
        assert heap.count() == 1
        assert heap.min() == 5

    def test_merge_heaps(self):
        """Test merging two heaps."""
        h1 = PairingHeap(3)
        h2 = PairingHeap(7)
        less_than = lambda a, b: a < b

        merged = h1.merge(h2, less_than)
        assert merged.min() == 3
        assert merged.count() == 2

    def test_merge_with_empty(self):
        """Test merging with empty heap."""
        h1 = PairingHeap(5)
        h_empty = PairingHeap(None)
        less_than = lambda a, b: a < b

        merged = h1.merge(h_empty, less_than)
        assert merged.min() == 5

    def test_insert(self):
        """Test inserting elements."""
        heap = PairingHeap(10)
        less_than = lambda a, b: a < b

        heap = heap.insert(5, less_than)
        assert heap.min() == 5

        heap = heap.insert(15, less_than)
        assert heap.min() == 5

    def test_remove_min(self):
        """Test removing minimum element."""
        heap = PairingHeap(5)
        less_than = lambda a, b: a < b

        heap = heap.insert(10, less_than)
        heap = heap.insert(3, less_than)

        assert heap.min() == 3
        heap = heap.remove_min(less_than)
        assert heap.min() == 5

    def test_heap_property(self):
        """Test that heap property is maintained."""
        heap = PairingHeap(10)
        less_than = lambda a, b: a < b

        heap = heap.insert(5, less_than)
        heap = heap.insert(20, less_than)
        heap = heap.insert(3, less_than)
        heap = heap.insert(15, less_than)

        assert heap.is_heap(less_than)

    def test_contains(self):
        """Test contains method."""
        heap1 = PairingHeap(5)
        heap2 = PairingHeap(10)
        less_than = lambda a, b: a < b

        merged = heap1.merge(heap2, less_than)
        assert merged.contains(heap1) or merged.contains(heap2)

    def test_for_each(self):
        """Test forEach iteration."""
        heap = PairingHeap(5)
        less_than = lambda a, b: a < b

        heap = heap.insert(10, less_than)
        heap = heap.insert(3, less_than)

        elements = []
        heap.for_each(lambda elem, h: elements.append(elem))

        assert 3 in elements
        assert 5 in elements
        assert 10 in elements
        assert len(elements) == 3

    def test_to_string(self):
        """Test string representation."""
        heap = PairingHeap(5)
        s = heap.to_string(str)
        assert "5" in s


class TestPriorityQueue:
    """Test cases for PriorityQueue class."""

    def test_create_empty_queue(self):
        """Test creating an empty queue."""
        pq = PriorityQueue(lambda a, b: a < b)
        assert pq.empty()
        assert pq.count() == 0
        assert pq.top() is None

    def test_push_and_top(self):
        """Test push and top operations."""
        pq = PriorityQueue(lambda a, b: a < b)

        pq.push(5)
        assert pq.top() == 5
        assert pq.count() == 1

        pq.push(3)
        assert pq.top() == 3
        assert pq.count() == 2

        pq.push(10)
        assert pq.top() == 3
        assert pq.count() == 3

    def test_push_multiple(self):
        """Test pushing multiple elements at once."""
        pq = PriorityQueue(lambda a, b: a < b)

        pq.push(5, 3, 10, 1, 7)
        assert pq.top() == 1
        assert pq.count() == 5

    def test_pop(self):
        """Test pop operation."""
        pq = PriorityQueue(lambda a, b: a < b)

        pq.push(5, 3, 10, 1, 7)

        assert pq.pop() == 1
        assert pq.pop() == 3
        assert pq.pop() == 5
        assert pq.pop() == 7
        assert pq.pop() == 10
        assert pq.empty()

    def test_pop_empty(self):
        """Test popping from empty queue."""
        pq = PriorityQueue(lambda a, b: a < b)
        assert pq.pop() is None

    def test_heap_property_maintained(self):
        """Test that heap property is maintained through operations."""
        pq = PriorityQueue(lambda a, b: a < b)

        pq.push(5, 3, 10, 1, 7, 15, 2, 8)
        assert pq.is_heap()

        pq.pop()
        assert pq.is_heap()

        pq.pop()
        assert pq.is_heap()

    def test_custom_comparison(self):
        """Test queue with custom comparison (max heap)."""
        # Max heap: larger values have higher priority
        pq = PriorityQueue(lambda a, b: a > b)

        pq.push(5, 3, 10, 1, 7)

        assert pq.pop() == 10
        assert pq.pop() == 7
        assert pq.pop() == 5

    def test_reduce_key(self):
        """Test reduce key operation."""
        pq = PriorityQueue(lambda a, b: a < b)

        # Push and keep reference to heap node
        node = pq.push(10)
        pq.push(5, 15)

        assert pq.top() == 5

        # Reduce key of node with value 10 to 2
        pq.reduce_key(node, 2)

        assert pq.top() == 2

    def test_reduce_key_with_callback(self):
        """Test reduce key with set_heap_node callback."""
        pq = PriorityQueue(lambda a, b: a < b)

        # Track which node contains which value
        node_map = {}

        def set_node(value, node):
            node_map[value] = node

        node = pq.push(10)
        node_map[10] = node
        pq.push(5, 15)

        # Reduce 10 to 2
        pq.reduce_key(node_map[10], 2, set_node)

        assert pq.top() == 2
        assert 2 in node_map

    def test_for_each(self):
        """Test forEach iteration."""
        pq = PriorityQueue(lambda a, b: a < b)

        pq.push(5, 3, 10, 1, 7)

        elements = []
        pq.for_each(lambda elem, h: elements.append(elem))

        assert len(elements) == 5
        assert all(x in elements for x in [1, 3, 5, 7, 10])

    def test_to_string(self):
        """Test string representation."""
        pq = PriorityQueue(lambda a, b: a < b)

        pq.push(5, 3, 10)

        s = pq.to_string(str)
        assert "3" in s  # Minimum should be in string

    def test_count(self):
        """Test count method."""
        pq = PriorityQueue(lambda a, b: a < b)

        assert pq.count() == 0

        pq.push(5)
        assert pq.count() == 1

        pq.push(3, 10)
        assert pq.count() == 3

        pq.pop()
        assert pq.count() == 2

    def test_large_dataset(self):
        """Test with larger dataset."""
        import random

        pq = PriorityQueue(lambda a, b: a < b)

        # Insert random numbers
        numbers = [random.randint(1, 1000) for _ in range(100)]
        for n in numbers:
            pq.push(n)

        assert pq.count() == 100
        assert pq.is_heap()

        # Pop all and verify order
        sorted_numbers = sorted(numbers)
        for expected in sorted_numbers:
            assert pq.pop() == expected

        assert pq.empty()
