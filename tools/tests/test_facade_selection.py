"""One building frontage may be fragmented, while parallel alley walls and corners must survive."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data/outdoor/scripts"))
from facade_fit import facade_explains_walls


def segment(a, b):
    return np.asarray(a, float), np.asarray(b, float), 2.5


facade = segment((0, 0), (10, 0))
fragments = [segment((0, .2), (5, 0)), segment((4, .3), (10, -.2))]


def test_crossing_frontage_fragments_can_merge():
    assert facade_explains_walls(facade, fragments, [s[:2] for s in fragments])


def test_parallel_wall_is_not_another_frontage_fragment():
    second = segment((0, 1), (10, 1))
    assert not facade_explains_walls(facade, fragments + [second], [s[:2] for s in fragments + [second]])


def test_corner_survives_even_if_direction_filter_would_remove_it():
    corner = segment((0, 0), (1.5, 2.5))
    assert not facade_explains_walls(facade, fragments, [s[:2] for s in fragments + [corner]])

