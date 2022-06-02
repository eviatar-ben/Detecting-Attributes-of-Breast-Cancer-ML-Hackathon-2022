def present_unique_values(df, col_name):
    print(df[col_name].unique())
    sums = 0
    for val in df[col_name].unique():
        num = (df[col_name] == val).sum()
        sums += num
        print(f"value: {val} has {num}")

    print(f"sums ={sums}")