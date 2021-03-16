from coding_the_matrix.Vec import Vec
from coding_the_matrix.matutil import (
    mat2rowdict,
    rowdict2mat,
    coldict2mat,
    mat2coldict,
    listlist2mat,
    lin_comb_mat_vec_mul,
    lin_comb_vec_mat_mul,
    dot_product_mat_vec_mul,
    dot_product_vec_mat_mul,
    m_v_mat_mat_mul,
    v_m_mat_mat_mul,
)
from coding_the_matrix.Mat import Mat
import pytest

from coding_the_matrix.image_mat_util import rotation
import numpy as np


def test_rowdict2mat():
    A = {0: Vec({0, 1}, {0: 1, 1: 2}), 1: Vec({0, 1}, {0: 3, 1: 4})}
    B = [Vec({0, 1}, {0: 1, 1: 2}), Vec({0, 1}, {0: 3, 1: 4})]
    assert mat2rowdict(rowdict2mat(A)) == A
    assert rowdict2mat(A) == Mat(
        ({0, 1}, {0, 1}), {(0, 1): 2, (1, 0): 3, (0, 0): 1, (1, 1): 4}
    )
    assert rowdict2mat(A) == rowdict2mat(B)


def test_coldict2mat():
    A = {0: Vec({0, 1}, {0: 1, 1: 2}), 1: Vec({0, 1}, {0: 3, 1: 4})}
    B = [Vec({0, 1}, {0: 1, 1: 2}), Vec({0, 1}, {0: 3, 1: 4})]
    assert mat2coldict(coldict2mat(A)) == A
    assert coldict2mat(A) == Mat(
        ({0, 1}, {0, 1}), {(0, 0): 1, (0, 1): 3, (1, 0): 2, (1, 1): 4}
    )
    assert coldict2mat(A) == coldict2mat(B)


def test_mat2coldict():
    M = Mat(
        ({0, 1, 2}, {0, 1}), {(0, 1): 1, (2, 0): 8, (1, 0): 4, (0, 0): 3, (2, 1): -2}
    )
    assert mat2coldict(M) == {
        0: Vec({0, 1, 2}, {0: 3, 1: 4, 2: 8}),
        1: Vec({0, 1, 2}, {0: 1, 1: 0, 2: -2}),
    }
    assert mat2coldict(Mat(({0, 1}, {0, 1}), {})) == {
        0: Vec({0, 1}, {0: 0, 1: 0}),
        1: Vec({0, 1}, {0: 0, 1: 0}),
    }


def test_mat2rowdict():
    M = Mat(
        ({0, 1, 2}, {0, 1}), {(0, 1): 1, (2, 0): 8, (1, 0): 4, (0, 0): 3, (2, 1): -2}
    )
    assert mat2rowdict(M) == {
        0: Vec({0, 1}, {0: 3, 1: 1}),
        1: Vec({0, 1}, {0: 4, 1: 0}),
        2: Vec({0, 1}, {0: 8, 1: -2}),
    }
    assert mat2rowdict(Mat(({0, 1}, {0, 1}), {})) == {
        0: Vec({0, 1}, {0: 0, 1: 0}),
        1: Vec({0, 1}, {0: 0, 1: 0}),
    }


def test_mat2rowdict_2():
    M = Mat(
        ({"radio", "sensor"}, {"memory", "CPU"}),
        {
            ("radio", "memory"): 3,
            ("radio", "CPU"): 1,
            ("sensor", "memory"): 2,
            ("sensor", "CPU"): 4,
        },
    )
    rowdict = mat2rowdict(M)
    assert set(rowdict.keys()) == M.D[0]
    assert rowdict["radio"] == Vec(M.D[1], {"memory": 3, "CPU": 1})


def test_listlist2mat():
    A = listlist2mat(
        (
            [
                [
                    3,
                    1,
                ],
                [4, 0],
                [8, -2],
            ]
        )
    )
    M = Mat(
        ({0, 1, 2}, {0, 1}), {(0, 1): 1, (2, 0): 8, (1, 0): 4, (0, 0): 3, (2, 1): -2}
    )
    assert A == M


def test_listlist2mat_rows():
    A = listlist2mat([[0, 1], [1, 0]], rows=["a", "b"])
    M = Mat(({"a", "b"}, {0, 1}), {("a", 1): 1, ("b", 0): 1})
    assert A == M


def test_listlist2mat_cols():
    A = listlist2mat([[0, 1], [1, 0]], cols=["c", "d"])
    M = Mat(({0, 1}, {"c", "d"}), {(0, "d"): 1, (1, "c"): 1})
    assert A == M


@pytest.fixture
def gnome_metal_matrix():
    D = {"metal", "concrete", "plastic", "water", "electricity"}
    v_gnome = Vec(
        D, {"concrete": 1.3, "plastic": 0.2, "water": 0.8, "electricity": 0.4}
    )
    v_hoop = Vec(D, {"plastic": 1.5, "water": 0.4, "electricity": 0.3})
    v_slinky = Vec(D, {"metal": 0.25, "water": 0.2, "electricity": 0.7})
    v_putty = Vec(D, {"plastic": 0.3, "water": 0.7, "electricity": 0.5})
    v_shooter = Vec(
        D, {"metal": 0.15, "plastic": 0.5, "water": 0.4, "electricity": 0.8}
    )
    rowdict = {
        "gnome": v_gnome,
        "hoop": v_hoop,
        "slinky": v_slinky,
        "putty": v_putty,
        "shooter": v_shooter,
    }
    #
    M = rowdict2mat(rowdict)
    return M


@pytest.fixture
def metal_vector():
    b = Vec(
        {"metal", "concrete", "water", "electricity", "plastic"},
        {
            "metal": 51.0,
            "concrete": 312.0,
            "water": 373.1,
            "electricity": 356.0,
            "plastic": 215.4,
        },
    )
    return b


@pytest.fixture
def gnome_vector():
    v = Vec(
        {"gnome", "hoop", "putty", "shooter", "slinky"},
        {
            "gnome": 240,
            "hoop": 55,
            "slinky": 150,
            "putty": 133,
            "shooter": 90,
        },
    )
    return v


def test_lin_comb_mat_vec_mul(gnome_metal_matrix, metal_vector, gnome_vector):
    """metal_gnome*gnome=metal"""
    actual = lin_comb_mat_vec_mul(gnome_metal_matrix.transpose(), gnome_vector)
    expected = metal_vector
    assert actual == expected


def test_lin_comb_vec_mat_mul(gnome_metal_matrix, metal_vector, gnome_vector):
    """gnome*gnome_metal=metal"""
    actual = lin_comb_vec_mat_mul(gnome_vector, gnome_metal_matrix)
    expected = metal_vector
    assert actual == expected


def test_dot_product_vec_mat_mul(gnome_metal_matrix, metal_vector, gnome_vector):
    """metal_gnome*gnome=metal"""
    actual = dot_product_mat_vec_mul(gnome_metal_matrix.transpose(), gnome_vector)
    expected = metal_vector
    assert actual == expected


def test_dot_product_mat_vec_mul(gnome_metal_matrix, metal_vector, gnome_vector):
    """gnome*gnome_metal=metal"""
    actual = dot_product_vec_mat_mul(gnome_vector, gnome_metal_matrix)
    expected = metal_vector
    assert actual == expected


def test_m_v_mat_mat_mul():
    A = rotation(np.pi / 4)
    expected = rotation(np.pi / 2)
    actual = m_v_mat_mat_mul(A, A)
    diff = actual - expected
    assert np.isclose(abs(diff), 0)


def test_v_m_mat_mat_mul():
    A = rotation(np.pi / 4)
    expected = rotation(np.pi / 2)
    actual = v_m_mat_mat_mul(A, A)
    diff = actual - expected
    assert np.isclose(abs(diff), 0)
