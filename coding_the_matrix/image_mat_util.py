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
from coding_the_matrix.matutil import mat2coldict, mat2rowdict, rowdict2mat
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image, ImageDraw


def im2colors(im: Image) -> Mat.Mat:
    """Get a color matrix from a Pillow image"""
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    r_dict = array_to_dict(r)
    col_labels = list(r_dict.keys())

    rowdict = {
        "r": Vec.Vec(set(col_labels), r_dict),
        "g": Vec.Vec(set(col_labels), array_to_dict(g)),
        "b": Vec.Vec(set(col_labels), array_to_dict(b)),
    }
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


def mat2im(colors: Mat.Mat, locations: Mat.Mat, density=6.0):
    """

    Parameters
    ----------
    colors :
    locations :
    density : number of plots per point

    Returns
    -------
    im
    """
    x_max, x_min, y_max, y_min = fig_size(locations)
    im = Image.new(
        mode="RGB",
        size=(int((x_max - x_min) * density), int((y_max - y_min) * density)),
        color="white",
    )
    d = ImageDraw.Draw(im)

    color_dict = mat2coldict(colors)
    location_dict = mat2coldict(locations)

    # make it square
    for top_left_point in sorted(color_dict):
        # color
        color = color_dict[top_left_point]
        hex_color = rgb_to_hex(color)

        # 4 points
        corner_index = four_corners(*top_left_point)
        corners = [location_dict[corner] for corner in corner_index]
        corner_tuples = [
            (corner["x"] * density, corner["y"] * density) for corner in corners
        ]

        # fill the polygon
        d.polygon(corner_tuples, fill=hex_color)

    return im


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
