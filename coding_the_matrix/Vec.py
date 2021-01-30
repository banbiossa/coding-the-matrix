class Vec:
    def __init__(self, domain: set, function: dict):
        """A vector is a function f from domain D to a field"""
        self.D = domain
        self.f = function

    def __repr__(self):
        return f"Vec({self.D}, {self.f})"

    def __getitem__(self, item):
        return self.f[item]

    def __setitem__(self, key, value):
        self.f[key] = value

    def __iter__(self):
        yield from self.f.values()
