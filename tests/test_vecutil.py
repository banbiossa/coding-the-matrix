from coding_the_matrix.vecutil import dictlist_helper


def test_dictlist_helper():
    inputs_dlist = [{"a": "apple", "b": "bear"}, {"a": 1, "b": 2}]
    actual = dictlist_helper(inputs_dlist, "a")
    expected = ["apple", 1]
    assert actual == expected
