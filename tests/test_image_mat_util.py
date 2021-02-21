from coding_the_matrix.image_mat_util import _color_int, corners
import pytest
from coding_the_matrix.Mat import Mat
from coding_the_matrix.matutil import listlist2mat
import itertools


@pytest.fixture
def gray_scale_squares():
    numbers = ["000111222", "012" * 3, "1" * 9]
    input_list = [[int(c) for c in number] for number in numbers]
    location = listlist2mat(
        input_list,
        rows=["x", "y", "u"],
        cols=[(x, y) for x, y in itertools.product(range(3), range(3))],
        # cols=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    )
    color = listlist2mat(
        # this becomes gray
        [[225, 125, 175, 75] for _ in range(3)],
        rows=["r", "g", "b"],
        cols=[(x, y) for x, y in itertools.product(range(2), range(2))],
    )
    return location, color


@pytest.mark.parametrize("test_input,expected", [(3.2, 3), (-1, 0), (257, 255)])
def test_color_int(test_input, expected):
    actual = _color_int(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([0, 0], [[0, 1, 1, 0], [0, 0, 1, 1]]),
        (
            [0, 1],
            [
                [0, 1, 1, 0],
                [
                    1,
                    1,
                    2,
                    2,
                ],
            ],
        ),
        ([1, 0], [[1, 2, 2, 1], [0, 0, 1, 1]]),
    ],
)
def test_corners(test_input, expected):
    actual = corners(*test_input)
    assert actual == expected
