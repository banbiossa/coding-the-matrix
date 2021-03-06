from coding_the_matrix.image_mat_util import (
    _color_int,
    four_corners,
    corners_to_list,
    rgb_to_hex,
    array_to_dict,
    fig_size,
)
import pytest
from coding_the_matrix.Mat import Mat
from coding_the_matrix.Vec import Vec
from coding_the_matrix.matutil import listlist2mat, mat2rowdict
from coding_the_matrix.vecutil import list2vec
import itertools
import numpy as np


@pytest.fixture
def gray_scale_squares():
    numbers = ["000111222", "012" * 3, "1" * 9]
    input_list = [[int(c) for c in number] for number in numbers]
    colors = listlist2mat(
        # this becomes gray
        [[225, 125, 175, 75] for _ in range(3)],
        rows=["r", "g", "b"],
        cols=[(x, y) for x, y in itertools.product(range(2), range(2))],
    )
    locations = listlist2mat(
        input_list,
        rows=["x", "y", "u"],
        cols=[(x, y) for x, y in itertools.product(range(3), range(3))],
        # cols=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    )
    return colors, locations


def test_fig_size(gray_scale_squares):
    colors, locations = gray_scale_squares
    actual = fig_size(locations)
    expected = (2, 0, 2, 0)
    assert actual == expected


@pytest.mark.parametrize("test_input,expected", [(3.2, 3), (-1, 0), (257, 255)])
def test_color_int(test_input, expected):
    actual = _color_int(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([0, 0], [(0, 0), (1, 0), (1, 1), (0, 1)]),
        ([0, 1], [(0, 1), (1, 1), (1, 2), (0, 2)]),
        ([1, 0], [(1, 0), (2, 0), (2, 1), (1, 1)]),
    ],
)
def test_four_corners(test_input, expected):
    actual = four_corners(*test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input,expected", [([0, 0, 0], "#000000"), ([0, 255, 0], "#00ff00")]
)
def test_rgb_to_hex(test_input, expected):
    color = Vec({"r", "g", "b"}, {c: test_input[i] for i, c in enumerate("rgb")})
    actual = rgb_to_hex(color)
    assert actual == expected


def test_corners_to_list():
    corners = mat2rowdict(listlist2mat([[0, 1, 1], [1, 0, 1]], cols=["x", "y", "u"]))
    actual = corners_to_list(corners.values())
    expected = [[0, 1], [1, 0], [1, 1]]
    assert actual == expected


def test_array_to_dict():
    array = np.array([[1, 0], [0, 1]])
    """
    1 0 
    0 1
    """
    actual = array_to_dict(array)
    expected = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1}
    assert actual == expected


def test_array_to_dict_2():
    array = np.array([[1, 2], [0, 1]])
    """
    1 2 
    0 1
    """
    actual = array_to_dict(array)
    expected = {(0, 0): 1, (0, 1): 0, (1, 0): 2, (1, 1): 1}
    assert actual == expected
