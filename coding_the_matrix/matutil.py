from coding_the_matrix import Vec, GF2, vecutil, Mat
import itertools
from tqdm import tqdm
from typing import Union


def rename_row_domain(M, R_to_r: Union[list[str], dict]):
    """Rename the row domain (R) for matrix

    Args:
        M ([Mat.Mat]): the matrix to transform
        R_to_r(list[str] or dict): the new row domain

    Returns: A new matrix with the renamed domain and function
    """
    R, C = M.D
    assert len(R) == len(R_to_r)  # same length domain
    assert len(set(R_to_r)) == len(R_to_r)  # no duplicates in domain

    # make a mapping of old_domain to new_domain
    if isinstance(R_to_r, list):
        R_to_r = {k: v for k, v in zip(R, R_to_r)}

    # make a rowdict and map the rowdict to new domain
    rowdict = mat2rowdict(M)
    new_rowdict = {R_to_r[k]: v for k, v in rowdict.items()}

    # rowdict to new matrix
    return rowdict2mat(new_rowdict, col_labels=C)


def efficient_rowdict2mat(rowdict):
    col_labels = value(rowdict).D
    M = Mat.Mat((set(keys(rowdict)), col_labels), {})
    for r in rowdict:
        for c in rowdict[r].f:
            M[r, c] = rowdict[r][c]
    return M


def identity(D, one):
    """Given a set D and the field's one, returns the DxD identity matrix
    e.g.:

    >>> identity({0,1,2}, 1)
    Mat(({0, 1, 2}, {0, 1, 2}), {(0, 0): 1, (1, 1): 1, (2, 2): 1})
    """
    return Mat.Mat((D, D), {(d, d): one for d in D})


def keys(d):
    """Given a dict, returns something that generates the keys; given a list,
    returns something that generates the indices.  Intended for coldict2mat and rowdict2mat.
    """
    return d.keys() if isinstance(d, dict) else range(len(d))


def value(d):
    """Given either a dict or a list, returns one of the values.
    Intended for coldict2mat and rowdict2mat.
    """
    return next(iter(d.values())) if isinstance(d, dict) else d[0]


def mat2rowdict(A):
    """Given a matrix, return a dictionary mapping row labels of A to rows of A
        e.g.:

    >>> M = Mat.Mat(({0, 1, 2}, {0, 1}), {(0, 1): 1, (2, 0): 8, (1, 0): 4, (0, 0): 3, (2, 1): -2})
    >>> mat2rowdict(M)
    {0: Vec({0, 1},{0: 3, 1: 1}), 1: Vec({0, 1},{0: 4, 1: 0}), 2: Vec({0, 1},{0: 8, 1: -2})}
    >>> mat2rowdict(Mat.Mat(({0,1},{0,1}),{}))
    {0: Vec({0, 1},{0: 0, 1: 0}), 1: Vec({0, 1},{0: 0, 1: 0})}
    """
    return {
        row: Vec.Vec(A.D[1], {col: A[row, col] for col in A.D[1]}) for row in A.D[0]
    }


def mat2coldict(A):
    """Given a matrix, return a dictionary mapping column labels of A to columns of A
    e.g.:
    >>> M = Mat.Mat(({0, 1, 2}, {0, 1}), {(0, 1): 1, (2, 0): 8, (1, 0): 4, (0, 0): 3, (2, 1): -2})
    >>> mat2coldict(M)
    {0: Vec({0, 1, 2},{0: 3, 1: 4, 2: 8}), 1: Vec({0, 1, 2},{0: 1, 1: 0, 2: -2})}
    >>> mat2coldict(Mat.Mat(({0,1},{0,1}),{}))
    {0: Vec({0, 1},{0: 0, 1: 0}), 1: Vec({0, 1},{0: 0, 1: 0})}
    """
    return {
        col: Vec.Vec(A.D[0], {row: A[row, col] for row in A.D[0]}) for col in A.D[1]
    }


def coldict2mat(coldict):
    """
    Given a dictionary or list whose values are Vecs, returns the Mat having these
    Vecs as its columns.  This is the inverse of mat2coldict.
    Assumes all the Vecs have the same label-set.
    Assumes coldict is nonempty.
    If coldict is a dictionary then its keys will be the column-labels of the Mat.
    If coldict is a list then {0...len(coldict)-1} will be the column-labels of the Mat.
    e.g.:

    >>> A = {0:Vec.Vec({0,1},{0:1,1:2}),1:Vec.Vec({0,1},{0:3,1:4})}
    >>> B = [Vec.Vec({0,1},{0:1,1:2}),Vec.Vec({0,1},{0:3,1:4})]
    >>> mat2coldict(coldict2mat(A)) == A
    True
    >>> coldict2mat(A)
    Mat(({0, 1}, {0, 1}), {(0, 0): 1, (0, 1): 3, (1, 0): 2, (1, 1): 4})
    >>> coldict2mat(A) == coldict2mat(B)
    True
    """
    row_labels = value(coldict).D
    return Mat.Mat(
        (row_labels, set(keys(coldict))),
        {(r, c): coldict[c][r] for c in keys(coldict) for r in row_labels},
    )


def rowdict2mat(rowdict, col_labels=None):
    """
    Given a dictionary or list whose values are Vecs, returns the Mat having these
    Vecs as its rows.  This is the inverse of mat2rowdict.
    Assumes all the Vecs have the same label-set.
    Assumes row_dict is nonempty.
    If rowdict is a dictionary then its keys will be the row-labels of the Mat.
    If rowdict is a list then {0...len(rowdict)-1} will be the row-labels of the Mat.
    e.g.:

    >>> A = {0:Vec.Vec({0,1},{0:1,1:2}),1:Vec.Vec({0,1},{0:3,1:4})}
    >>> B = [Vec.Vec({0,1},{0:1,1:2}),Vec.Vec({0,1},{0:3,1:4})]
    >>> mat2rowdict(rowdict2mat(A)) == A
    True
    >>> rowdict2mat(A)
    Mat(({0, 1}, {0, 1}), {(0, 1): 2, (1, 0): 3, (0, 0): 1, (1, 1): 4})
    >>> rowdict2mat(A) == rowdict2mat(B)
    True
    """
    if col_labels is None:
        col_labels = value(rowdict).D
    return Mat.Mat(
        (list(keys(rowdict)), col_labels),
        {(r, c): rowdict[r][c] for r in keys(rowdict) for c in col_labels},
    )


def listlist2mat(
    L,
    rows=None,
    cols=None,
):
    """Given a list of lists of field elements, return a matrix.

    If rows or cols are empty, the ith row consists
    of the elements of the ith list.  The row-labels are {0...len(L)}, and the
    column-labels are {0...len(L[0])}
    >>> A=listlist2mat([[10,20,30,40],[50,60,70,80]])
    >>> print(A)
        0   1   2   3
    0  10  20  30  40
    1  50  60  70  80
    """
    if rows is None:
        rows = list(range(len(L)))
    if cols is None:
        cols = list(range(len(L[0])))
    # assert isinstance(rows, list), f"Must be list not {type(rows)}, order is important."
    # assert isinstance(cols, list), f"Must be list not {type(cols)}, order is important."
    assert len(L) == len(rows)
    assert len(L[0]) == len(cols)
    return Mat.Mat(
        (rows, cols),
        {
            (r, c): L[ir][ic]
            for (ir, r) in enumerate(rows)
            for (ic, c) in enumerate(cols)
        },
    )


def submatrix(M, rows, cols):
    return Mat.Mat(
        (M.D[0] & rows, M.D[1] & cols),
        {(r, c): val for (r, c), val in M.f.items() if r in rows and c in cols},
    )


def button_vectors(n):
    """Button vectors of n*n"""
    D = {(x, y) for x, y in itertools.product(range(n), range(n))}
    vecdict = {
        (i, j): Vec.Vec(
            D,
            dict(
                [((x, j), GF2.one) for x in range(max(i - 1, 0), min(i + 2, n))]
                + [((i, y), GF2.one) for y in range(max(j - 1, 0), min(j + 2, n))]
            ),
        )
        for (i, j) in D
    }
    return vecdict


def is_triangular(M):
    R_orig, C_orig = M.D
    # for a certain R -> C mapping
    for R, C in tqdm(
        itertools.product(
            itertools.permutations(R_orig), itertools.permutations(C_orig)
        )
    ):
        # if f == 0 for all j > i, return True
        if all(
            [
                M[(R[i], C[j])] == 0
                for (i, j) in itertools.product(range(len(R)), range(len(C)))
                if i > j
            ]
        ):
            return True, R, C
    return False, None, None


def lin_comb_mat_vec_mul(M, v):
    """M*v, using linear-combination definition
    v[k] is the only operation allowed on v.
    Use the linear combination definition.

    Args:
        M: matrix
        v: vector
    """
    R, C = M.D
    assert v.D == C
    result = vecutil.zero_vec(R)
    for col in C:
        m_col = Vec.Vec(R, {r: M[(r, col)] for r in R})
        result += m_col * v[col]
    return result


def lin_comb_vec_mat_mul(v, M):
    """linear combination version of v*M"""
    R, C = M.D
    assert v.D == R
    result = vecutil.zero_vec(C)
    for row in R:
        r_col = Vec.Vec(C, {c: M[(row, c)] for c in C})
        result += r_col * v[row]
    return result


def dot_product_mat_vec_mul(M, v):
    """Dot product version of M *v
    RxC * C = Rx1
    """
    R, C = M.D
    assert v.D == C
    return Vec.Vec(R, {r: sum(M[(r, c)] * v[c] for c in C) for r in R})


def dot_product_vec_mat_mul(v, M):
    """Dot product version of v * M

    Parameters
    ----------
    v : 1 * R
    M : R * C

    Returns
    -------
    v * M = 1 * C
    """
    R, C = M.D
    assert v.D == R
    return Vec.Vec(C, {c: sum(M[(r, c)] * v[r] for r in R) for c in C})


def m_v_mat_mat_mul(A, B):
    """M*v only """
    coldict = {}
    for k, col in mat2coldict(B).items():
        coldict[k] = A * col
    return coldict2mat(coldict)


def v_m_mat_mat_mul(A, B):
    """v*M only"""
    rowdict = {}
    for k, row in mat2rowdict(A).items():
        rowdict[k] = row * B
    return rowdict2mat(rowdict)
