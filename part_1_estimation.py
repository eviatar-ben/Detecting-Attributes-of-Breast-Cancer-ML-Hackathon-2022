import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from main import parse_features


# python3 evaluate_part_1.py --gold=./splited_datasets/y_test_base1.csv --pred=./part1_baseline_pred.csv
def estimate_1():
    features = pd.read_csv("splited_datasets/X_train_base1.csv")

    df, num_imp, ord_imp, encoder = parse_features(features)

    labels = pd.read_csv("splited_datasets/y_train_base1.csv")

    model = LinearRegression()
    model.fit(df, labels)

    test_features = pd.read_csv("splited_datasets/X_test_base1.csv")

    X_test, num_imp, ord_imp, encoder = parse_features(test_features, num_imp, ord_imp, encoder)
    #
    # pred = model.predict(X_test)
    #
    # pred[pred < 0] = 0
    #
    # pd.DataFrame(pred, columns=["אבחנה-Tumor size"]).to_csv("part1_baseline_pred.csv", index=False)

    df = pd.concat([df, X_test])
    labels = pd.concat([labels, pd.read_csv("splited_datasets/y_test_base1.csv")])

    print("################## RidgeCV ##################")
    model1 = RidgeCV()
    print(cross_val_score(model1, df, labels, scoring="neg_mean_squared_error", cv=KFold(shuffle=True)))

    print("################## LassCV ##################")
    model2 = LassoCV()
    print(cross_val_score(model2, df, labels.to_numpy().T[0], scoring="neg_mean_squared_error", cv=KFold(shuffle=True)))

    print("################## DecisionTree ##################")
    model3 = DecisionTreeRegressor()
    print(cross_val_score(model3, df, labels, scoring="neg_mean_squared_error", cv=KFold(shuffle=True)))

    print("################## KneighborsRegressor ##################")
    model4 = KNeighborsRegressor()
    print(cross_val_score(model4, df, labels, scoring="neg_mean_squared_error", cv=KFold(shuffle=True)))

    print("################## average of DecisionTree and  Knn##################")
    model5 = KNeighborsRegressor()
    model6 = DecisionTreeRegressor()
    print(cross_val_score(model5, df, labels, scoring="neg_mean_squared_error", cv=KFold(shuffle=True)))


if __name__ == '__main__':
    for i in range(5):
        np.random.seed(i)
        print(f"################## starting seed {i} ##################")
        estimate_1()
