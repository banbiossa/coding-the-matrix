import pandas as pd


class Mat:
    def __init__(self, labels, function):
        self.D = labels
        self.f = function

    def __str__(self):
        R, C = self.D
        row_dict = {r: [self.f[(r, c)] for c in C] for r in R}
        return pd.DataFrame.from_dict(row_dict, orient="index").to_string()
