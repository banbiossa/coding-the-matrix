from coding_the_matrix.solve import solve, get_max_row
from coding_the_matrix.Vec import Vec
from coding_the_matrix.GF2 import one
from coding_the_matrix import matutil
import numpy as np


def test_get_max_row():
    rowdict = {
        0: Vec({0, 1}, {0: 0, 1: 0}),
        1: Vec({0, 1}, {0: 1, 1: 0}),
    }
    row, vec = get_max_row(0, rowdict)
    assert row == 1
    assert vec == Vec({0, 1}, {0: 1, 1: 0})


def test_get_max_row_gf2():
    rowdict = {
        0: Vec({0, 1}, {0: 0, 1: 0}),
        1: Vec({0, 1}, {0: one, 1: 0}),
    }
    row, vec = get_max_row(0, rowdict)
    assert row == 1
    assert vec == Vec({0, 1}, {0: one, 1: 0})


def test_solve():
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
    M = matutil.rowdict2mat(rowdict)
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
    A = M.transpose()
    x = solve(A, b)
    residual = b - M * x
    assert np.isclose(residual * residual, 0)


def test_buttons():
    # 5 * 5 の行列
    vecdict = matutil.button_vectors(5)
    B = matutil.rowdict2mat(vecdict)
    b = matutil.value(vecdict)
    s = Vec(b.D, {(2, 2): one})
    sol = solve(B, s)
    assert B * sol == s
