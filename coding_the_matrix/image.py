import png
import numbers
import collections


def _color_int(col):
    """
    Round color coordinate to nearest int and clamp to [0, 255]

    Parameters
    ----------
    col :

    Returns
    -------

    """
    return max(min(round(col), 255), 0)


def _boxed2flat(row):
    """
    utility conversions, between boxed pixel and flat pixel formats
    the png library uses flat, we use boxed.

    Parameters
    ----------
    row :

    Returns
    -------

    """
    return [_color_int(x) for box in row for x in box]


def _flat2boxed(row):
    """
    Note we skip every 4th element, thus eliminating the alpha channel

    Parameters
    ----------
    row :

    Returns
    -------

    """
    return [tuple(row[i : i + 3]) for i in range(0, len(row), 4)]


def isgray(image):
    """tests whether the image is grayscale"""
    col = image[0][0]
    if isinstance(col, numbers.Number):
        return True
    elif isinstance(col, collections.Iterable) and len(col) == 3:
        return False
    else:
        raise TypeError("Unrecognized image type")


def color2gray(image):
    """Converts a color image to grayscale

    we use HDTV grayscale conversion as per https://en.wikipedia.org/wiki/Grayscale
    """
    image = [[x for x in row] for row in image]
    return [
        [int(0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) for p in row]
        for row in image
    ]


def gray2color(image):
    """ Converts a grayscale image to color """
    return [[(p, p, p) for p in row] for row in image]


def rgbsplit(image):
    """ Converts an RGB image to a 3-element list of grayscale images, one for each color channel"""
    return [[[pixel[i] for pixel in row] for row in image] for i in (0, 1, 2)]


def rgpsplice(R, G, B):
    return [
        [(R[row][col], G[row][col], B[row][col]) for col in range(len(R[0]))]
        for row in range(len(R))
    ]


def file2image(path):
    """Reads an image into a list of lists of pixel values (tuples with
    three values). This is a color image."""
    (w, h, p, m) = png.Reader(filename=path).asRGBA()  # force RGB and alpha
    return [_flat2boxed(r) for r in p]


def image2file(image, path):
    """Writes an image in list of lists format to a file. Will work with
    either color or grayscale."""
    if isgray(image):
        img = gray2color(image)
    else:
        img = image
    with open(path, "wb") as f:
        png.Writer(width=len(image[0]), height=len(image)).write(
            f, [_boxed2flat(r) for r in img]
        )
