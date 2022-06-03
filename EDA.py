import pandas as pd
from sklearn.model_selection import train_test_split


def eda():
    labels1 = pd.read_csv("task 2/train.labels.1.csv")

    full_fea = pd.read_csv("task 2/train.feats.csv").drop([" Form Name", "User Name"], axis=1).drop_duplicates()

    labels1 = labels1.loc[full_fea.index]

    X_train, X_test, y_train, y_test = train_test_split(full_fea, labels1)

    y_test.to_csv("splited_datasets/y_test_one.csv", index=False)
    X_test.to_csv("splited_datasets/X_test_one.csv", index=False)
    X_train.to_csv("splited_datasets/X_train_one.csv", index=False)
    y_train.to_csv("splited_datasets/y_train_one.csv", index=False)


if __name__ == '__main__':
    eda()
