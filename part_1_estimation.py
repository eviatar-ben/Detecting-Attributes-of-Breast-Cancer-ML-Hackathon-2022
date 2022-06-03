import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from main import parse_features


# python3 evaluate_part_1.py --gold=./splited_datasets/y_test_base1.csv --pred=./part1_baseline_pred.csv
def estimate_1(data, label):
    forset_loss = np.mean(
        cross_val_score(RandomForestRegressor(max_features=0.75, max_samples=0.75), data, label.to_numpy().T[0],
                        scoring="neg_mean_squared_error", cv=KFold(shuffle=True)))

    return forset_loss


if __name__ == '__main__':
    losses = []
    features = pd.read_csv("splited_datasets/X_train_one.csv", parse_dates=[
        "אבחנה-Diagnosis date",
        "אבחנה-Surgery date1",
        "אבחנה-Surgery date2",
        "אבחנה-Surgery date3"
    ], infer_datetime_format=True, dayfirst=True)

    df, num_imp, ord_imp, encoder = parse_features(features)

    df = df.drop_duplicates()

    labels = pd.read_csv("splited_datasets/y_train_one.csv")

    labels = labels.loc[df.index]

    for i in range(5):
        np.random.seed(i)
        print(f"################## starting seed {i} ##################")
        round_loss = estimate_1(df, labels)
        print(round_loss)
        losses.append(round_loss)
    print("########### overall average loss over iterations ###########")

    values = np.mean(losses, axis=0)
    print(values)
