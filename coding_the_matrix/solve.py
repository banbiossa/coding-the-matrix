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
    return row, vec


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
    rowdict = matutil.mat2rowdict(M)
    used = set()  # 結果をいれる

    # select col to operate on
    R, C = M.D
    for col in C:
        # get row where col is max value
        row, rowvec = get_max_row(col, rowdict, used=used)
        assert row not in used
        # rowdict.pop(row)
        if rowvec[col] == 0:
            logger.error(f"{rowvec=} has zero at {row=}, {col=}")
            continue
        used.add(row)

        # update yourself (to one)
        coef = 1 / rowvec[col]
        rowvec *= coef
        assert abs(rowvec[col] - 1) < eps
        b[row] *= coef
        rowdict[row] = rowvec

        # iterate over the rows
        for other, othervec in rowdict.items():
            if row == other:
                continue

            # coef
            coef = othervec[col] / rowvec[col]

            # update rowvec
            othervec -= coef * rowvec
            assert othervec[col] == 0
            rowdict[other] = othervec

            # update b
            b[other] -= coef * b[row]

            A = matutil.rowdict2mat(rowdict)
            logger.debug(f"{A}")

    # map the vectors to the right domain
    # return b, A
    mapper = {k[0]: k[1] for k, v in A.f.items() if v != 0}
    if len(mapper) != len(C):
        logger.error(f"{mapper=} not length {len(C)=}.")

    x = Vec.Vec(b.D, {mapper[k]: v for k, v in b.f.items()})
    return x
