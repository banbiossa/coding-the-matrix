"""
The original `image_mat_util.py` and `png.py` was very complex (lot's of glue code).
The complexity comes mostly from supporting old versions of python (like 2.2 and 3.5), and
 the fact that the original author was reluctant to rely on external libraries
 (maybe as this could make the installation process harder).

I hope to make a simpler API leveraging numpy and Pillow (less glue code).
The functionality of each the original module is relatively simple,
and we are in Python 3.9 with a rich ecosystem for dealing with png and arrays
(mostly numpy and matplotlib). I hope I can load the work to these libraries and
make a relatively simple API on our handling of pngs and arrays.
"""

import itertools
from math import ceil
from typing import Tuple
from coding_the_matrix import Vec
from coding_the_matrix import Mat
from coding_the_matrix.matutil import mat2coldict, mat2rowdict, rowdict2mat
import matplotlib.pyplot as plt
import io
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from functools import reduce
import operator


def reduce_end_mul(values, end):
    """Reduce multiply but specifies the end (the original matrix)"""
    if isinstance(values, Mat.Mat):
        values = [values]
    assert isinstance(values, list)
    assert isinstance(end, Mat.Mat)
    values.append(end)
    start = values.pop(0)
    return reduce(operator.mul, values, start)


def show(
    colors,
    locations,
    col_mat=None,
    loc_mat=None,
):
    """Can take lists for loc_mat and col_mat"""
    if col_mat is None:
        col_mat = scale_color(1, 1, 1)
    if loc_mat is None:
        loc_mat = identity()

    im = mat2im(reduce_end_mul(col_mat, colors), reduce_end_mul(loc_mat, locations))
    return plt.imshow(im)


def to_transformation(funcs: dict) -> Mat.Mat:
    """Make a dict of dicts to a color or location transformation matrix

    Args:
        funcs: a dict of values to be made into a matrix
            e.g., {'x': {'x': 1}}
            e.g., {'r': {'r': 1, 'g': 2}, 'g': {'g': 1}}
            Takes {'r', 'g', 'b'} or {'x', 'y', 'u'} domain dicts
    """
    assert isinstance(funcs, dict)

    def _find_D() -> list[str]:
        # get the domain, either xyu or rgb
        for D in [["x", "y", "u"], ["r", "g", "b"]]:
            if any([key in funcs for key in D]):
                return D
        raise ValueError(f"Can't find domain for {funcs.keys()}")

    # get the D that matches
    D = _find_D()
    # make a 3D dict of Vecs
    rowdict: dict[str, Vec.Vec] = {
        key: Vec.Vec(set(D), funcs.get(key, {})) for key in D
    }
    return rowdict2mat(rowdict, col_labels=D)


def grayscale():
    """Return grayscale rgb 77r/256, 151g/256, 28b/256"""
    funcs = {
        key: {"r": 77 / 256, "g": 151 / 256, "b": 28 / 256} for key in {"r", "g", "b"}
    }
    return to_transformation(funcs)


def scale_color(r, g, b):
    """Scale the colors"""
    funcs = {
        "r": {"r": r},
        "g": {"g": g},
        "b": {"b": b},
    }
    return to_transformation(funcs)


def identity() -> Mat.Mat:
    """Returns an identity matrix for location vectors"""
    funcs = dict(
        x={"x": 1},
        y={"y": 1},
        u={"u": 1},
    )
    return to_transformation(funcs)


def translation(alpha, beta) -> Mat.Mat:
    """(x, y) -> (x+alpha, y+beta) on a (x, y, u) representation"""
    funcs = dict(
        x={"x": 1, "u": alpha},
        y={"y": 1, "u": beta},
        u={"u": 1},
    )
    return to_transformation(funcs)


def scale(alpha, beta) -> Mat.Mat:
    """Scale the matrix [x, y] -> [x*alpha, y*beta]"""
    funcs = dict(
        x={"x": alpha},
        y={"y": beta},
        u={"u": 1},
    )
    return to_transformation(funcs)


def rotation(theta) -> Mat.Mat:
    """Rotate the matrix [x, y] -> [cos(theta)*x - sin(theta)*y, sin(theta)*x + cos(theta)*y]"""
    funcs = dict(
        x={"x": np.cos(theta), "y": -np.sin(theta)},
        y={"x": np.sin(theta), "y": np.cos(theta)},
        u={"u": 1},
    )
    return to_transformation(funcs)


def rotation_about(theta, x, y) -> Mat.Mat:
    """Rotate around x, y"""
    return translation(x, y) * rotation(theta) * translation(-x, -y)


def reflect_y() -> Mat.Mat:
    """(x, y) -> (-x, y)"""
    funcs = dict(
        x={"x": -1},
        y={"y": 1},
        u={"u": 1},
    )
    return to_transformation(funcs)


def reflect_x() -> Mat.Mat:
    """(x, y) -> (-x, y)"""
    funcs = dict(
        x={"x": 1},
        y={"y": -1},
        u={"u": 1},
    )
    return to_transformation(funcs)


def im2mat(im: Image) -> Tuple[Mat.Mat, Mat.Mat]:
    """Returns color matrix and location matrix from Pillow image"""
    return im2colors(im), im2locations(im)


def im2colors(im: Image) -> Mat.Mat:
    """Get a color matrix from a Pillow image
    Maps the rgb, square -> value
    """
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    r_dict = array_to_dict(r)
    col_labels = list(r_dict.keys())

    rowdict = {
        "r": Vec.Vec(set(col_labels), r_dict),
        "g": Vec.Vec(set(col_labels), array_to_dict(g)),
        "b": Vec.Vec(set(col_labels), array_to_dict(b)),
    }
    return rowdict2mat(rowdict, col_labels=col_labels)


def im2locations(im: Image) -> Mat.Mat:
    """Get a location matrix from a pillow Image
    Locations is a (x, y) -> (x, y, 1) matrix that is 1 larger than the original color matrix
    Conceptually the corners of the matrix
    """
    x, y, _ = im.shape
    col_labels = [(i, j) for i, j in itertools.product(range(x + 1), range(y + 1))]

    rowdict = dict(
        x=Vec.Vec(set(col_labels), function={key: key[0] for key in col_labels}),
        y=Vec.Vec(set(col_labels), function={key: key[1] for key in col_labels}),
        u=Vec.Vec(set(col_labels), function={key: 1 for key in col_labels}),
    )
    return rowdict2mat(rowdict, col_labels=col_labels)


def array_to_dict(array: np.array) -> dict:
    """Unpack a 2d array to {(row, col): val}
    The tricky part is that a row, col => j, i
    This is because when a col increases, x increases
    and row increases, y increases

       x ->> x increase left
    y
    |
    |
    y increase down

    e.g., For the matrix
    0 1
    2 3

    0: 0,0
    1: is (row, col) = (0, 1), (x,y) = (1, 0)
    2: is (row, col) = (1, 0), (x,y) = (0, 1)
    3: 1,1
    """
    assert array.ndim == 2
    return {(j, i): val for (i, row) in enumerate(array) for (j, val) in enumerate(row)}


def init_blank_image(locations: Mat.Mat, density=1.0):
    """Get a blank image (canvas?) to plot the locations on.
    A square shape that just fits the locations

    Parameters
    ----------
    locations : locations matrix
    density:

    Returns
    -------
    An image
    """
    x_max, x_min, y_max, y_min = fig_size(locations)
    im = Image.new(
        mode="RGB",
        size=(ceil(x_max * density), ceil(y_max * density)),
        color="white",
    )
    return im


def mat2im(colors: Mat.Mat, locations: Mat.Mat, im: Image = None, density=1.0):
    """

    Parameters
    ----------
    colors :
    locations :
    im: the image to draw on
    density:

    Returns
    -------
    im
    """
    if im is None:
        im = init_blank_image(locations, density=density)
    d = ImageDraw.Draw(im)

    color_dict = mat2coldict(colors)
    location_dict = mat2coldict(locations)

    # if all are minus, raise an Error
    if not any(vec["x"] > 0 and vec["y"] > 0 for vec in location_dict.values()):
        raise RuntimeError(
            "All values are minus, need at least one to print. Consider translation(+x, +y)"
        )

    # make it square
    for top_left_point in sorted(color_dict):
        # color
        color = color_dict[top_left_point]
        hex_color = rgb_to_hex(color)

        # 4 points
        corner_index = four_corners(*top_left_point)
        corners = [location_dict[corner] for corner in corner_index]
        corner_tuples = [
            (int(corner["x"] * density), int(corner["y"] * density))
            for corner in corners
        ]

        # fill the polygon
        d.polygon(corner_tuples, fill=hex_color)

    return im


def fig_max(all_locations):
    """get the maximum from all locations"""
    return (
        pd.DataFrame(
            [fig_size(loc) for loc in all_locations],
            columns=["x_max", "x_min", "y_max", "y_min"],
        )
        .max()
        .loc[["x_max", "y_max"]]
        .to_dict()
        .values()
    )


def show_colors(colors: Mat.Mat, locations: Mat.Mat, height=1.0):
    """from a colors matrix and locations matrix, return the figure

    Parameters
    ----------
    colors :
    locations :
    height : figheight

    Returns
    -------
    figure

    """
    assert colors.D[0] == {"r", "g", "b"}
    assert locations.D[0] == {"x", "y", "u"}
    color_dict = mat2coldict(colors)
    location_dict = mat2coldict(locations)

    # make it square
    fig, ax = plt.subplots(figsize=(height, height))
    plt.tight_layout(pad=0)
    plt.margins(x=0, y=0)
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


def fig_size(locations):
    """return x_max, x_min, y_max, y_min"""
    rowdict = mat2rowdict(locations)
    return rowdict["x"].max, rowdict["x"].min, rowdict["y"].max, rowdict["y"].min


def fig2arr(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    with io.BytesIO() as buf:
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return np.array(img)


def _fig_to_array(fig) -> np.array:
    """
    A quick way to get the numpy representation from a fig.
    Isn't complete, as this relies on how the figure is rendered.
    """
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
    if type(n).__module__ == np.__name__:
        raise TypeError(f"{n} shouldn't be numpy but an int.")
    return hex(n)[2:].zfill(2)


def _color_int(col: float) -> int:
    as_int = max(min(int(round(col)), 255), 0)
    return as_int


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
