import pandas as pd
from sklearn.model_selection import train_test_split


def eda():
    labels1 = pd.read_csv("train.labels.1.csv")

    full_fea = pd.read_csv("train.feats.csv")

    X_train, X_test, y_train, y_test = train_test_split(full_fea, labels1)

    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_train, y_train)

    y_test.to_csv("splited_datasets/y_test1")
    X_test.to_csv("splited_datasets/X_test1")
    X_train_base.to_csv("splited_datasets/X_train_base1")
    y_train_base.to_csv("splited_datasets/y_train_base1")
    X_test_base.to_csv("splited_datasets/X_test_base1")
    y_test_base.to_csv("splited_datasets/y_test_base1")


if __name__ == '__main__':
    eda()
