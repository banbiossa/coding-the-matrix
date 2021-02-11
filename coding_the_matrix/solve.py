"""
掃き出し法の実装を試みる

"""
from coding_the_matrix import Mat, Vec, matutil
from typing import Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_max_row(col, rowdict, used=None) -> Tuple[str, Vec.Vec]:
    """get row from rowdict where col is max"""
    rowdict = rowdict.copy()
    if used is not None:
        [rowdict.pop(key) for key in used]
    row, vec = sorted(rowdict.items(), key=lambda x: abs(x[1].f[col]))[-1]
    assert vec[col] != 0
    return row, vec


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
    b = b.copy()
    rowdict = matutil.mat2rowdict(M)
    used = set()  # 結果をいれる

    # select col to operate on
    R, C = M.D
    for col in C:
        # get row where col is max value
        row, rowvec = get_max_row(col, rowdict, used=used)
        assert row not in used
        used.add(row)
        # rowdict.pop(row)

        # update yourself (to one)
        coef = 1 / rowvec[col]
        rowvec *= coef
        assert np.isclose(rowvec[col], 1)
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
    mapper = {k[0]: k[1] for k, v in A.f.items() if v != 0}
    assert len(mapper) == 5
    x = Vec.Vec(set(mapper.values()), {mapper[k]: v for k, v in b.f.items()})
    return x


if __name__ == "__main__":
    D = {"metal", "concrete", "plastic", "water", "electricity"}
    v_gnome = Vec.Vec(
        D, {"concrete": 1.3, "plastic": 0.2, "water": 0.8, "electricity": 0.4}
    )
    v_hoop = Vec.Vec(D, {"plastic": 1.5, "water": 0.4, "electricity": 0.3})
    v_slinky = Vec.Vec(D, {"metal": 0.25, "water": 0.2, "electricity": 0.7})
    v_putty = Vec.Vec(D, {"plastic": 0.3, "water": 0.7, "electricity": 0.5})
    v_shooter = Vec.Vec(
        D, {"metal": 0.15, "plastic": 0.5, "water": 0.4, "electricity": 0.8}
    )
    rowdict = {
        "gnome": v_gnome,
        "hoop": v_hoop,
        "slinky": v_slinky,
        "putty": v_putty,
        "shooter": v_shooter,
    }
    M = matutil.rowdict2mat(rowdict)
    b = Vec.Vec(
        {"metal", "concrete", "water", "electricity", "plastic"},
        {
            "metal": 51.0,
            "concrete": 312.0,
            "water": 373.1,
            "electricity": 356.0,
            "plastic": 215.4,
        },
    )
    A = M.transpose()
    x = solve(A, b)
    residual = b - x * M
    assert np.isclose(residual * residual, 0)
    print(x)
