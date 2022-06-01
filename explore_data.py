import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    features = pd.read_csv("train.feats.csv")
    labels = pd.read_csv("train.labels.0.csv")
    X_train, X_test, y_train, y_test = train_test_split(features, labels)
    X_test.to_csv("./splited_datasets/features_test_0.csv")
    y_test.to_csv("./splited_datasets/labels_test_0.csv")
    X_train_base, X_base, y_train_base, y_base = train_test_split(X_train, y_train)
    X_train_base.to_csv("./splited_datasets/features_train_base_0.csv")
    X_base.to_csv("./splited_datasets/features_test_base_0.csv")
    y_train_base.to_csv("./splited_datasets/labels_train_base_0.csv")
    y_base.to_csv("./splited_datasets/labels_test_base_0.csv")
