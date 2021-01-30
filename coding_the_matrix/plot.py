from numbers import Number

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def parse_point(pt):
    """Parse a point"""
    if isinstance(pt, Number):
        return pt.real, pt.imag

    if isinstance(pt, (tuple, list)):
        if len(pt) != 2:
            raise ValueError(f"{pt} must be length 2")
        return pt

    raise ValueError(f"can only parse tuple, list or Number, not {type(pt)}")


def parse_list(L: list) -> pd.DataFrame:
    """Parse a list of points to a dataframe"""
    all_points = []
    for pt in L:
        x, y = parse_point(pt)
        all_points.append(
            {
                "x": x,
                "y": y,
            }.copy()
        )
    return pd.DataFrame(all_points)


def plot(L: list, ax=None, *args, **kwargs):
    """

    Parameters
    ----------
    L : list of points
    ax

    Returns
    -------
    ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    df = parse_list(L)
    sns.scatterplot(x="x", y="y", data=df, ax=ax, *args, **kwargs)
    return ax
