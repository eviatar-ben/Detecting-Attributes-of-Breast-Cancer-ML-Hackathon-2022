""" Usage:
    <file-name> (part1 | part2 | part3) --train-x=TRAIN_X --train-y=TRAIN_Y [--test-x=TEST_X]

Options:
  --help                           Show this message and exit
"""
from sklearn.impute import SimpleImputer
from docopt import docopt
from pathlib import Path
import logging
import pandas as pd
from typing import Tuple, Iterable

def load_data(train_X_fn: Path, train_y_fn: Path):
    features = pd.read_csv(train_X_fn, parse_dates=[
        "אבחנה-Diagnosis date",
        "אבחנה-Surgery date1",
        "אבחנה-Surgery date2",
        "אבחנה-Surgery date3",
        "surgery before or after-Activity date"
    ], infer_datetime_format=True, dayfirst=True)
    labels = pd.read_csv(train_y_fn)
    full_data = pd.concat([features, labels])
    return full_data


def handle_numerical(df: pd.DataFrame) -> Iterable[SimpleImputer]:
    numerical_categories = [
        "אבחנה-Tumor depth",
        "אבחנה-Tumor width",
        "אבחנה-Surgery sum",
        "אבחנה-Positive nodes",
        "אבחנה-Nodes exam",
        "אבחנה-Age",
    ]
    median_imputer = SimpleImputer(strategy="median")
    df["אבחנה-Age"] = median_imputer.fit_transform(df[["אבחנה-Age"]])
    # df["אבחנה-Age"] = median_imputer.transform(df["אבחנה-Age"])
    return [median_imputer]


if __name__ == '__main__':
    args = docopt(__doc__)
    # print(args)
    if args["part1"]:
        train_X_fn = Path(args["--train-x"])
        train_y_fn = Path(args["--train-y"])
        # test_X_fn = Path(args["--test-x"])

        df = load_data(train_X_fn, train_y_fn)
        handle_numerical(df)
        a = df.describe()
        # head = df.head(1000)
        # a = head.describe()
        # for colname, colval in df.iteritems():
        #     print(colname)
        #     print(pd.unique(head[colname]))


