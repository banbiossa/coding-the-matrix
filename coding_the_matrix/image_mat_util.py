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


# Round color coordinate to nearest int and clamp to [0, 255]
def _hex(n):
    """zero filled hex"""
    return hex(n)[2:].zfill(2)


def _color_int(col: float) -> int:
    return max(min(round(col), 255), 0)


def rgb_to_hex(rgb):
    """3 value rgb tuple to hex"""
    assert len(rgb) == 3
    rgb = [_color_int(color) for color in rgb]
    return "#" + "".join([_hex(n) for n in rgb])


def corners(x, y) -> list[list]:
    """Return the corners, starting from `x, y` counter clockwise."""
    point_order = [(0, 0), (1, 0), (1, 1), (0, 1)]
    corner_order = [(x + dx, y + dy) for dx, dy in point_order]
    return [[corner[i] for corner in corner_order] for i in (0, 1)]


# utility conversions, between boxed pixel and flat pixel formats
# the png library uses flat, we use boxed.
def _boxed2flat(row):
    return [_color_int(x) for box in row for x in box]


def _flat2boxed(row):
    # Note we skip every 4th element, thus eliminating the alpha channel
    return [tuple(row[i : i + 3]) for i in range(0, len(row), 4)]
