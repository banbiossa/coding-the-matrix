from coding_the_matrix.Vec import Vec


def zero_vec(D):
    """outputs an instance of Vec representing a D-vector
    all of whose entries have value zero
    """
    return Vec(D, {key: 0 for key in D})


def scalar_mul(v, alpha):
    """
    Args:
        v: vector
        alpha: scalar

    Returns:
        a new instance of Vec that represents the scalar-vector product alpha times v
    """
    return Vec(v.D, {key: alpha * value for key, value in v.f.items()})


def list_dot(u, v):
    return sum([i * j for i, j in zip(u, v)])


def list2vec(L):
    """Vec with domain {0, 1, ..., len(L)} and v[i] = L[i]"""
    return Vec({i for i in range(len(L))}, {i: L[i] for i in range(len(L))})
