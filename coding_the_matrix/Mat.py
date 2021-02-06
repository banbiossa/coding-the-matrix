import pandas as pd
from .Vec import Vec
from .matutil import mat2coldict, mat2rowdict


def vec_mul_mat(u, M):
    """vec * matrix multiplication"""
    assert u.D == M.D[0]
    # get a row representation of matrix
    return Vec(M.D[1], {k: u * vec for k, vec in mat2coldict(M).items()})


def mat_mul_vec(M, u):
    """matrix * vec multiplication"""
    assert M.D[1] == u.D
    return Vec(M.D[0], {k: vec * u for k, vec in mat2rowdict(M).items()})


class Mat:
    def __init__(self, labels, function):
        self.D = labels
        self.f = function

    def __getitem__(self, value):
        assert isinstance(value, tuple)
        assert len(value) == 2
        return self.f.get(value, 0)

    def __mul__(self, other):
        """M * u"""
        if isinstance(other, Vec):
            return mat_mul_vec(self, other)
        # matrix multiplication necessary
        return NotImplemented

    def __rmul__(self, other):
        """u * M"""
        if isinstance(other, Vec):
            return vec_mul_mat(other, self)
        return NotImplemented

    def __str__(self):
        R, C = self.D
        row_dict = {r: [self.f.get((r, c), 0) for c in C] for r in R}
        df = pd.DataFrame.from_dict(row_dict, orient="index")
        df.columns = C
        return df.to_string()
