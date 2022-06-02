import numpy as np
import pandas as pd


def present_unique_values(df, col_name):
    print(df[col_name].unique())
    sums = 0
    for val in df[col_name].unique():
        num = (df[col_name] == val).sum()
        sums += num
        print(f"value: {val} has {num}")

    num = len(df[df[col_name].isnull()].index.tolist())
    sums += num
    print(f"value: {np.nan} has {num}")

    print(f"values number that are not numpy.nan ={sums}")
