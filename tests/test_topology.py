from quroute import from_edges, grid_topology, linear_topology, ring_topology, trivial_layout


def test_linear():
    cm = linear_topology(4)
    assert cm.size() == 4
    assert cm.distance(0, 3) == 3


def test_ring_wraps():
    cm = ring_topology(4)
    # 4 个比特的环里,0 和 3 相邻
    assert cm.distance(0, 3) == 1


def test_grid():
    cm = grid_topology(2, 3)
    assert cm.size() == 6
    assert cm.distance(0, 5) == 3


def test_from_edges_symmetric():
    cm = from_edges([(0, 1), (1, 2)])
    assert cm.distance(0, 2) == 2
    assert cm.distance(2, 0) == 2  # 无向


def test_trivial_layout():
    assert trivial_layout(3) == {0: 0, 1: 1, 2: 2}
