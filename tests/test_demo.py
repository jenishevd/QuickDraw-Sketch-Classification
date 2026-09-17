import numpy as np
import pytest

from demo import has_sufficient_ink, is_confident


def test_has_sufficient_ink_rejects_a_blank_canvas():
    assert not has_sufficient_ink(np.full((100, 100), 255, dtype=np.uint8))


def test_has_sufficient_ink_accepts_a_dense_mark():
    canvas = np.full((100, 100), 255, dtype=np.uint8)
    canvas[30:70, 30:70] = 0

    assert has_sufficient_ink(canvas)


@pytest.mark.parametrize(
    ("probs", "expected"),
    [
        (np.full(12, 1 / 12), False),
        (np.array([0.45] + [0.05] * 11), True),
    ],
)
def test_is_confident_uses_the_top_probability(probs, expected):
    assert is_confident(probs) is expected
