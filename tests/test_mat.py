import itertools
from coding_the_matrix.matutil import (
    listlist2mat,
)
from coding_the_matrix.Mat import Mat, mat_mul_mat
import numpy as np
import pytest


def test_getitem():
    M = Mat(({0}, {0}), {(0, 0): 1})
    with pytest.raises(AssertionError):
        a = M[0]


def test_mat_mul_mat():
    # E * E = E
    U = listlist2mat([[1, 0], [0, 1]])
    V = listlist2mat([[1, 0], [0, 1]])
    A = mat_mul_mat(U, V)
    assert A == U


def test_mat_mul_mat_2():
    # 45 * 2 = 90
    _sqrt2 = 1 / np.sqrt(2)
    U = listlist2mat([[_sqrt2, -_sqrt2], [_sqrt2, _sqrt2]])
    V = listlist2mat([[0, -1], [1, 0]])
    A = mat_mul_mat(U, U)
    for i, j in itertools.product(*A.D):
        assert np.isclose(A[i, j], V[i, j])


def test_mat_mul():
    # 45 * 2 = 90
    _sqrt2 = 1 / np.sqrt(2)
    U = listlist2mat([[_sqrt2, -_sqrt2], [_sqrt2, _sqrt2]])
    V = listlist2mat([[0, -1], [1, 0]])
    A = U * U
    for i, j in itertools.product(*A.D):
        assert np.isclose(A[i, j], V[i, j])


def test_mat_mul_2():
    # 135 * 2 = 270
    _sqrt2 = 1 / np.sqrt(2)
    U = listlist2mat([[-_sqrt2, -_sqrt2], [_sqrt2, -_sqrt2]])
    V = listlist2mat([[0, 1], [-1, 0]])
    A = U * U
    for i, j in itertools.product(*A.D):
        assert np.isclose(A[i, j], V[i, j])


def test_mat_mul_num():
    U = listlist2mat([[1, 0], [0, 1]])
    num = 2
    actual = U * num
    expected = listlist2mat([[2, 0], [0, 2]])
    assert actual == expected

    actual = U * 3
    expected = listlist2mat([[2, 0], [0, 2]])
    assert actual != expected

    actual = num * U
    expected = listlist2mat([[2, 0], [0, 2]])
    assert actual == expected
