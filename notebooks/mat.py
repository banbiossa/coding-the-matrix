import pandas as pd
from coding_the_matrix import Vec
from coding_the_matrix.matutil import mat2coldict, mat2rowdict


def vec_mul_mat(u, M):
    """vec * matrix multiplication"""
    assert u.D == M.D[0]
    # get a row representation of matrix
    return Vec.Vec(M.D[1], {k: u * vec for k, vec in mat2coldict(M).items()})


def mat_mul_vec(M, u):
    """matrix * vec multiplication"""
    assert M.D[1] == u.D
    return Vec.Vec(M.D[0], {k: vec * u for k, vec in mat2rowdict(M).items()})


class Mat:
    def __init__(self, labels, function):
        self.D = labels
        self.f = function

    def __repr__(self):
        D0 = set(sorted(self.D[0]))
        D1 = set(sorted(self.D[1]))
        f = {k: self.f[k] for k in sorted(self.f)}
        return "Mat(({}, {}), {})".format(D0, D1, f)

    def __eq__(self, other):
        return self.D[0] == other.D[0] and self.D[1] == other.D[1] and self.f == other.f

    def __getitem__(self, value):
        assert isinstance(value, tuple)
        assert len(value) == 2
        return self.f.get(value, 0)

    def __setitem__(self, key, value):
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert key[0] in self.D[0]
        assert key[1] in self.D[1]
        self.f[key] = value

    def __mul__(self, other):
        """M * u"""
        if isinstance(other, Vec.Vec):
            return mat_mul_vec(self, other)
        # matrix multiplication necessary
        return NotImplemented

    def __rmul__(self, other):
        """u * M"""
        if isinstance(other, Vec.Vec):
            return vec_mul_mat(other, self)
        return NotImplemented

    def __str__(self):
        R, C = self.D
        row_dict = {r: [self.f.get((r, c), 0) for c in C] for r in R}
        df = pd.DataFrame.from_dict(row_dict, orient="index")
        df.columns = C
        return df.to_string()

    def transpose(self):
        R, C = self.D
        D = (C, R)
        f = {(c, r): v for (r, c), v in self.f.items()}
        return Mat(D, f)
