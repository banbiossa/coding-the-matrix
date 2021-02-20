from coding_the_matrix.image_mat_util import _color_int
import pytest
from coding_the_matrix.Mat import Mat


def test_color_int():
    pass


@pytest.fixture
def gray_scale_squares():
    location = Mat(
        (
            {"x", "y", "u"},
            {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)},
        ),
    )
