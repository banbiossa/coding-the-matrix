from .Vec import Vec


def zero_vec(D):
    """outputs an instance of Vec representing a D-vector
    all of whose entries have value zero
    """
    return Vec(D, {key: 0 for key in D})


def setitem(v, d, val):
    v.f[d] = val


def getitem(v, d):
    return v.f[d] if d in v.f else 0


def scalar_mul(v, alpha):
    """
    Args:
        v: vector
        alpha: scalar

    Returns:
        a new instance of Vec that represents the scalar-vector product alpha times v
    """
    return Vec(v.D, {key: alpha * value for key, value in v.f.items()})


def add(u, v):
    """
    Returns:
        Vec that is the sum of u and v
    """
    return Vec(u.D, {key: u.f.get(key, 0) + v.f.get(key, 0) for key in u.D})


def neg(v):
    return Vec(v.D, {key: -v.f.get(key, 0) for key in u.D})


def list_dot(u, v):
    return sum([i * j for i, j in zip(u, v)])


def mul(u, v):
    return Vec(u.D, {d: value * getitem(v, d) for d, value in u.f.items()})


def dot(u, v):
    return sum(mul(u, v))


def list2vec(L):
    """Vec with domain {0, 1, ..., len(L)} and v[i] = L[i]"""
    return Vec({i for i in range(len(L))}, {i: L[i] for i in range(len(L))})
