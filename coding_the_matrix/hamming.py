"""
Hamming error correction codes
"""
from coding_the_matrix.GF2 import one
from coding_the_matrix.Mat import Mat
from coding_the_matrix.Vec import Vec
from coding_the_matrix.matutil import listlist2mat, mat2coldict, coldict2mat
from coding_the_matrix.vecutil import list2vec, zero_vec

G = listlist2mat(
    [
        [one, 0, one, one],
        [one, one, 0, one],
        [0, 0, 0, one],
        [one, one, one, 0],
        [0, 0, one, 0],
        [0, one, 0, 0],
        [one, 0, 0, 0],
    ]
)
H = listlist2mat(
    [
        [0, 0, 0, one, one, one, one],
        [0, one, one, 0, 0, one, one],
        [one, 0, one, 0, one, 0, one],
    ]
)
R = listlist2mat(
    [
        [0, 0, 0, 0, 0, 0, one],
        [0, 0, 0, 0, 0, one, 0],
        [0, 0, 0, 0, one, 0, 0],
        [0, 0, one, 0, 0, 0, 0],
    ]
)


def find_error(error_syndrome: Vec) -> Vec:
    """

    Parameters
    ----------
    error_syndrome :H (3,7) * C_tilda (7,1) = H * e (3, 1)
        must be a 3 vector

    Returns
    -------
    a 7 vector, 1 corresponding to the error

    """
    assert error_syndrome.D == {0, 1, 2}
    coldict = mat2coldict(H)  # 7 of 3 vectors
    for i, col in coldict.items():
        if error_syndrome == col:
            zeros = [0 for i in range(7)]
            zeros[i] = one
            return list2vec(zeros)
    return zero_vec(set(range(7)))


def find_error_matrix(S: Mat):
    """
    Args:
        S: columns are error syndromes (3, n)
    Returns:
        a matrix whose column is the error corresponding to the column of S
        a (7, n) matrix
    """
    assert S.D[0] == {0, 1, 2}
    coldict = mat2coldict(S)  # n of 3 vectors
    errors = {k: find_error(v) for k, v in coldict.items()}
    return coldict2mat(errors)


def correct(A):
    """
    Args:
        errored codeword matrix
    Returns:
        a matrix are valid codewords (4*n)
    """
    decoded = H * A
    error = find_error_matrix(decoded)
    corrected = A - error
    message = R * corrected
    return message
