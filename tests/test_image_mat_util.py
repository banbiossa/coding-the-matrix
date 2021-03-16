import matplotlib

from coding_the_matrix.image_mat_util import (
    _color_int,
    _hex,
    four_corners,
    corners_to_list,
    rgb_to_hex,
    array_to_dict,
    fig_size,
    show,
    scale_color,
    reduce_end_mul,
    scale,
    translation,
    reflect_x,
    reflect_y,
    rotation,
    identity,
    rotation_about,
    reflect_about,
)
import pytest
from coding_the_matrix.Vec import Vec
from coding_the_matrix.Mat import Mat
from coding_the_matrix.matutil import listlist2mat, mat2rowdict
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


def test_hex():
    actual = _hex(0)
    expected = "00"
    assert actual == expected

    actual = _hex(255)
    expected = "ff"
    assert actual == expected

    # hex only takes ints
    with pytest.raises(TypeError):
        _hex(255.1)

    # hex only takes ints
    with pytest.raises(TypeError):
        _hex(np.float32(322))


@pytest.mark.parametrize(
    "test_input,expected", [(3.2, 3), (-1, 0), (257, 255), (np.float64(1.1), 1)]
)
def test_color_int(test_input, expected):
    actual = _color_int(test_input)
    assert type(actual) == int
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
    "test_input,expected",
    [([0, 0, 0], "#000000"), ([0, 255, 0], "#00ff00"), ([0.1, 255, 0], "#00ff00")],
)
def test_rgb_to_hex(test_input, expected):
    color = Vec({"r", "g", "b"}, {c: test_input[i] for i, c in enumerate("rgb")})
    actual = rgb_to_hex(color)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input,test_output", [([1, 0], [1, 0]), ([0, 1], [0, -1])]
)
def test_reflect_x(test_input, test_output):
    test_vec_equal(reflect_x(), test_input, test_output)


@pytest.mark.parametrize(
    "test_input,test_output", [([1, 0], [-1, 0]), ([0, 1], [0, 1])]
)
def test_reflect_y(test_input, test_output):
    test_vec_equal(reflect_y(), test_input, test_output)


@pytest.mark.parametrize(
    "theta,test_input,test_output",
    [
        (0, [1, 0], [1, 0]),
        (np.pi, [1, 1], [-1, -1]),
        (np.pi / 2, [1, 0], [0, 1]),
        (np.pi / 6, [1, 0], [np.sqrt(3) / 2, 1 / 2]),
    ],
)
def test_rotation(theta, test_input, test_output):
    test_vec_nearly_equal(rotation(theta), test_input, test_output)


@pytest.mark.parametrize(
    "alpha,beta,test_input,test_output",
    [
        (0, 0, [1, 0], [1, 0]),
        (1, 0, [1, 1], [2, 1]),
        (np.pi, 2, [1, 0], [1 + np.pi, 2]),
        (0, np.pi / 6, [0, 1 / 2], [0, 1 / 2 + np.pi / 6]),
    ],
)
def test_translation(alpha, beta, test_input, test_output):
    test_vec_equal(translation(alpha, beta), test_input, test_output)


@pytest.mark.parametrize(
    "test_input,test_output",
    [
        ([1, 0], [1, 0]),
        ([1, 1], [1, 1]),
        ([0, 1 / 2], [0, 1 / 2]),
    ],
)
def test_identity(test_input, test_output):
    test_vec_equal(identity(), test_input, test_output)


@pytest.mark.parametrize(
    "alpha,beta,test_input,test_output",
    [
        (1, 1, [1, 0], [1, 0]),
        (1, 2, [1, 1], [1, 2]),
        (0, 1 / 2, [1, 1], [0, 1 / 2]),
    ],
)
def test_scale(alpha, beta, test_input, test_output):
    test_vec_equal(scale(alpha, beta), test_input, test_output)


@pytest.mark.parametrize(
    "theta,x,y,test_input,test_output",
    [
        (0, 1, 1, [1, 0], [1, 0]),
        (np.pi / 2, 1, 1, [1, 0], [2, 1]),
    ],
)
def test_rotation_about(theta, x, y, test_input, test_output):
    test_vec_nearly_equal(rotation_about(theta, x, y), test_input, test_output)


@pytest.mark.parametrize(
    "x1,y1,x2,y2,test_input,test_output",
    [
        (0, 0, 1, 1, [1, 0], [0, 1]),
        (-1, 0, 1, 0, [0, 1], [0, -1]),
        (0, 1, 0, -1, [1, 0], [-1, 0]),
    ],
)
def test_reflect_about(x1, y1, x2, y2, test_input, test_output):
    test_vec_nearly_equal(reflect_about(x1, y1, x2, y2), test_input, test_output)


@pytest.mark.skip("This isn't a test but a utility")
def test_vec_equal(transformation: Mat, test_input, test_output):
    """Utility function to test vector equalness"""
    D = {"x", "y", "u"}
    point = Vec(D, {"x": test_input[0], "y": test_input[1], "u": 1})
    actual = transformation * point
    expected = Vec(D, {"x": test_output[0], "y": test_output[1], "u": 1})
    assert actual == expected


@pytest.mark.skip("This isn't a test but a utility")
def test_vec_nearly_equal(transformation: Mat, test_input, test_output):
    """Utility function to test vector near equalness (floating point differences)"""
    D = {"x", "y", "u"}
    point = Vec(D, {"x": test_input[0], "y": test_input[1], "u": 1})
    actual = transformation * point
    expected = Vec(D, {"x": test_output[0], "y": test_output[1], "u": 1})
    diff = actual - expected
    assert diff * diff < 1e-7


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


def test_scale_colors(gray_scale_squares):
    colors, locations = gray_scale_squares

    img = show(colors, locations, col_mat=scale_color(1, 1, 2))
    assert isinstance(img, matplotlib.image.AxesImage)


def test_scale_colors_2(gray_scale_squares):
    colors, locations = gray_scale_squares
    img = show(colors, locations, col_mat=scale_color(1 / 2, 4, 2))
    assert isinstance(img, matplotlib.image.AxesImage)


def test_reduce_end_mul(gray_scale_squares):
    colors, locations = gray_scale_squares
    loc_transformations = [scale(3, 2), translation(2, 1)]
    mat = reduce_end_mul(loc_transformations, locations)
    assert isinstance(mat, Mat)
