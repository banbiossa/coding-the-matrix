from coding_the_matrix.Vec import Vec
from coding_the_matrix.matutil import (
    mat2rowdict,
    rowdict2mat,
    coldict2mat,
    mat2coldict,
    listlist2mat,
)
from coding_the_matrix.Mat import Mat


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
