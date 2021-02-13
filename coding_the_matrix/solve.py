"""
掃き出し法の実装を試みる

"""
from coding_the_matrix import Mat, Vec, matutil, GF2
from typing import Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_max_row(col, rowdict, used=None) -> Tuple[str, Vec.Vec]:
    """
    get row from rowdict where col is max
    """
    rowdict = rowdict.copy()
    if used is not None:
        [rowdict.pop(key) for key in used]
    row, vec = sorted(rowdict.items(), key=lambda x: abs(x[1].f[col]))[-1]
    if vec[col] == 0:
        logger.error(f"No max row for {rowdict=}, {col=}")
    return row, vec.copy()


def solve(M: Mat.Mat, b: Vec.Vec, eps=1e-9) -> Vec.Vec:
    """Solve M * x = b (return x)
    try: 掃き出し法

    Parameters
    ----------
    M :
    b :
    eps:

    Returns
    -------
    x

    """
    b = b.copy()
    M = M.copy()
    rowdict = matutil.mat2rowdict(M)
    assert M.D[0] == b.D
    used = dict()  # 結果をいれる
    # 行->列の対応. x はMの列のベクトルだが、途中計算は全てbの行方向のベクトルのなかで行われるため

    # select col to operate on
    R, C = M.D
    for col in C:
        # get row where col is max value
        row, rowvec = get_max_row(col, rowdict, used=used)
        assert row not in used
        # rowdict.pop(row)
        used.update({row: col}.copy())

        # if 0, don't use it, and delete it from everything
        if rowvec[col] == 0:
            # if there are non zero items, probably  unsolvable
            nonzero = {k: v for k, v in rowvec.f.items() if v != 0}
            if len(nonzero) != 0:
                logger.error(f"{rowvec=} unsolvable {row=}, {col=}, {nonzero=}")

            # delete the col value for everything
            for other, othervec in rowdict.items():
                othervec = othervec.copy()
                othervec[col] = 0
                rowdict[other] = othervec.copy()
            continue  # to the next item

        # update yourself (to one)
        coef = 1 / rowvec[col]
        rowvec *= coef
        assert abs(rowvec[col] - 1) < eps
        b[row] *= coef
        rowdict[row] = rowvec.copy()

        # iterate over the rows
        for other, othervec in rowdict.items():
            if row == other:
                continue
            othervec = othervec.copy()

            # coef
            coef = othervec[col] / rowvec[col]

            # update rowvec
            othervec -= coef * rowvec
            assert othervec[col] == 0
            rowdict[other] = othervec.copy()

            # update b
            b[other] -= coef * b[row]

            # update the unit matrix (for A^-1)
            # todo: add the actual code

            A = matutil.rowdict2mat(rowdict)
            logger.debug(f"{A}")

    # map the vectors to the right domain
    # map the row -> value to col -> value as the final vector
    assert set(used.keys()) == A.D[0]
    assert set(used.values()) == A.D[1]
    x = Vec.Vec(A.D[1], {used[k]: v for k, v in b.f.items()})

    return x
