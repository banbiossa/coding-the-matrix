import pandas as pd
from coding_the_matrix import Vec
from coding_the_matrix import matutil
from numbers import Number


def vec_mul_mat(u, M):
    """vec * matrix multiplication"""
    assert u.D == M.D[0]
    # get a row representation of matrix
    return Vec.Vec(M.D[1], {k: u * vec for k, vec in matutil.mat2coldict(M).items()})


def mat_mul_vec(M, u):
    """matrix * vec multiplication"""
    assert M.D[1] == u.D
    return Vec.Vec(M.D[0], {k: vec * u for k, vec in matutil.mat2rowdict(M).items()})


def mat_mul_mat(U, V):
    """matrix * matrix multiplication"""
    assert U.D[1] == V.D[0]
    rowdict = matutil.mat2rowdict(U)
    for key, row in rowdict.items():
        rowdict[key] = row * V
    return matutil.rowdict2mat(rowdict, col_labels=V.original_labels[1])


def mat_mul_num(M, num):
    """matrix number multiplication (use the underlying vec)"""
    rowdict = matutil.mat2rowdict(M)
    for key, row_vec in rowdict.items():
        rowdict[key] = row_vec * num
    return matutil.rowdict2mat(rowdict, col_labels=M.original_labels[1])


class Mat:
    def __init__(self, labels, function):
        self._original_labels = [item.copy() for item in labels]
        labels = [set(label) for label in labels]
        assert len(labels) == 2
        assert all([isinstance(d, set) for d in labels])
        assert all([isinstance(k, tuple) and len(k) == 2 for k in function.keys()])
        assert all([i in labels[0] and j in labels[1] for (i, j) in function.keys()])
        self.D = labels
        self.f = function

    @property
    def original_labels(self):
        return self._original_labels

    @property
    def shape(self):
        return len(self.D[0]), len(self.D[1])

    @property
    def max(self):
        return max(self.f.values())

    @property
    def min(self):
        return min(self.f.values())

    def copy(self):
        return self.__class__(self.D, self.f.copy())

    def __repr__(self):
        return "Mat({}, {})".format(self.D, self.f)

    def __neg__(self):
        return self.__class__(self._original_labels, {k: -v for k, v in self.f.items()})

    def __eq__(self, other) -> bool:
        """

        Parameters
        ----------
        other : Mat
        """
        same_class = isinstance(other, Mat)
        same_D = self.D[0] == other.D[0] and self.D[1] == other.D[1]
        same_f = self._sparse_f() == other._sparse_f()
        return same_D and same_f and same_class

    def _sparse_f(self):
        return {k: v for k, v in self.f.items() if v != 0}

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
        if isinstance(other, self.__class__):
            return mat_mul_mat(self, other)
        if isinstance(other, Number):
            return mat_mul_num(self, other)
        raise NotImplementedError(f"{type(self)} and {type(other)}")

    def __rmul__(self, other):
        """u * M"""
        if isinstance(other, Vec.Vec):
            return vec_mul_mat(other, self)
        if isinstance(other, Number):
            return mat_mul_num(self, other)
        raise NotImplementedError(f"{type(self)} and {type(other)}")

    def __add__(self, other):
        # add each item if other is matrix
        if isinstance(other, Mat):
            assert other.D == self.D
            return Mat(
                self.D,
                {
                    k: (self[k] + other[k])
                    for k in set(self.f.keys()) | set(other.f.keys())
                },
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Mat):
            assert other.D == self.D
            return Mat(
                self.D,
                {
                    k: (self[k] - other[k])
                    for k in set(self.f.keys()) | set(other.f.keys())
                },
            )
        return NotImplemented

    def __str__(self):
        R, C = self._original_labels
        row_dict = {r: [self.f.get((r, c), 0) for c in C] for r in R}
        df = pd.DataFrame.from_dict(row_dict, orient="index")
        df.columns = C
        return df.to_string()

    def to_pandas(self):
        R, C = self.D
        row_dict = {r: [self.f.get((r, c), 0) for c in C] for r in R}
        df = pd.DataFrame.from_dict(row_dict, orient="index")
        df.columns = C
        return df

    def transpose(self):
        R, C = self.D
        D = (C, R)
        f = {(c, r): v for (r, c), v in self.f.items()}
        return Mat(D, f)

    def pprint(self, rows=None, cols=None):
        """Reorder the matrix (useful for triangular matrices)"""
        df = self.to_pandas()
        R, C = self.D
        if rows is not None:
            assert set(rows) == R
            df = df.loc[rows, :]
        if cols is not None:
            assert set(cols) == C
            df = df.loc[:, cols]
        return df
