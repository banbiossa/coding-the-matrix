"""
掃き出し法の実装を試みる

"""
from coding_the_matrix import Mat, Vec, matutil


def solve(M: Mat.Mat, b: Vec.Vec) -> Vec.Vec:
    """Solve M * x = b (return x)
    try: 掃き出し法

    Parameters
    ----------
    M :
    b :

    Returns
    -------
    x

    # 2 * 2 の行列
    >>> M = matutil.coldict2mat({0:Vec.Vec({0,1},{0:1,1:2}),1:Vec.Vec({0,1},{0:3,1:4})})
    >>> M
    Mat(({0, 1}, {0, 1}), {(0, 0): 1, (0, 1): 3, (1, 0): 2, (1, 1): 4})
    """
    pass
