"""
The original `image_mat_util.py` and `png.py` is way too complex.
This comes (mostly) from supporting old versions of python (like 2.2 and 3.5), and
 that the original author was reluctant to rely on external libraries
 (as this would indeed make the install process a bit more of a hustle).

I hope to make a simpler API for the same ends here.
The functionality of each the original module is relatively simple,
and we are in Python 3.9 with a rich ecosystem for dealing with png and arrays
(mostly numpy and matplotlib). I hope I can load the work to these libraries and
make a relatively simple API on our handling of pngs and arrays.
"""

from coding_the_matrix import Vec
from coding_the_matrix import Mat
from coding_the_matrix.matutil import mat2coldict
import matplotlib.pyplot as plt
import io
import numpy as np


def show_colors(colors: Mat.Mat, locations: Mat.Mat):
    """from a colors matrix and locations matrix, return the figure

    Parameters
    ----------
    colors :
    locations :

    Returns
    -------
    figure

    """
    color_dict = mat2coldict(colors)
    location_dict = mat2coldict(locations)

    fig, ax = plt.subplots()
    for top_left_point in sorted(color_dict):
        # color
        color = color_dict[top_left_point]
        hex_color = rgb_to_hex(color)

        # 4 points
        corner_index = four_corners(*top_left_point)
        corners = [location_dict[corner] for corner in corner_index]
        x, y, u = corners_to_list(corners)
        ax.fill(x, y, color=hex_color)
    return fig


def fig_to_array(fig) -> np.array:
    """
    A quick way to get the numpy representation from a fig.
    Isn't complete, as this relies on how the figure is rendered.
    """
    fig.tight_layout(pad=0)
    with io.BytesIO() as io_buf:
        fig.savefig(io_buf, format="raw")
        io_buf.seek(0)
        img_arr = np.reshape(
            np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
        )
    return img_arr


# Round color coordinate to nearest int and clamp to [0, 255]
def _hex(n):
    """zero filled hex"""
    return hex(n)[2:].zfill(2)


def _color_int(col: float) -> int:
    return max(min(round(col), 255), 0)


def rgb_to_hex(rgb: Vec.Vec):
    """3 value rgb Vec to hex"""
    assert len(rgb) == 3
    rgb_list = [_color_int(rgb[color]) for color in "rgb"]
    return "#" + "".join([_hex(n) for n in rgb_list])


def four_corners(x, y) -> list[tuple]:
    """Return the corners, starting from `x, y` counter clockwise."""
    point_order = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return [(x + dx, y + dy) for dx, dy in point_order]


def corners_to_list(corners: list[Vec.Vec]):
    """[Vecs] to [(x,...), (y,...)]"""
    return [[corner[idx] for corner in corners] for idx in ["x", "y", "u"]]


# utility conversions, between boxed pixel and flat pixel formats
# the png library uses flat, we use boxed.
def _boxed2flat(row):
    return [_color_int(x) for box in row for x in box]


def _flat2boxed(row):
    # Note we skip every 4th element, thus eliminating the alpha channel
    return [tuple(row[i : i + 3]) for i in range(0, len(row), 4)]
