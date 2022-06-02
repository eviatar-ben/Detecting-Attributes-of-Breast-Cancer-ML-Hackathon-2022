""" Usage:
    <file-name> (part1 | part2 | part3) --train-x=TRAIN_X --train-y=TRAIN_Y [--test-x=TEST_X]

Options:
  --help                           Show this message and exit
"""
from pandas import CategoricalDtype
from sklearn.impute import SimpleImputer
from docopt import docopt
from pathlib import Path
import logging
import pandas as pd
from typing import Tuple, Iterable
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from preprocessor import *
from explore_data import *

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
    full_data = full_data.loc[:, ~full_data.columns.str.contains('^Unnamed')]
    full_data.reset_index(inplace=True, drop=True)
    return full_data


def handle_numerical(df: pd.DataFrame) -> Iterable[SimpleImputer]:
    numerical_categories = [
        "אבחנה-Tumor depth",  # TODO: Drop?
        "אבחנה-Tumor width",  # TODO: Drop?
        "אבחנה-Surgery sum",  # TODO: fill using dates?
        "אבחנה-Positive nodes",  # todo:
        "אבחנה-Nodes exam",  # TODO:
        "אבחנה-Age",
    ]

    df['אבחנה-Surgery sum'].mask(
        (df['אבחנה-Surgery sum'].isna()),
        (df['אבחנה-Surgery date1'].notna()) +
        (df['אבחנה-Surgery date2'].notna()) +
        (df['אבחנה-Surgery date3'].notna()),
        inplace=True
    )

    df['אבחנה-Nodes exam'].fillna(0, inplace=True)
    df['אבחנה-Positive nodes'].mask(
        (df["אבחנה-Positive nodes"].isna()), df['אבחנה-Nodes exam'],
        inplace=True
    )
    median_imputer = SimpleImputer(strategy="median")
    df["אבחנה-Age"] = median_imputer.fit_transform(df[["אבחנה-Age"]])
    # df["אבחנה-Age"] = median_imputer.transform(df["אבחנה-Age"])
    return [median_imputer]


def handle_ordered_categories(df: pd.DataFrame) -> Iterable[SimpleImputer]:
    ordered_categories = [
        "אבחנה-Basic stage",  # TODO: switch to unordered - ignore Null?
        "אבחנה-Histopatological degree",  # TODO: Gx and null in diff columns?
        "אבחנה-Lymphatic penetration",  # TODO: LI vs L1? Null?
        "אבחנה-M -metastases mark (TNM)",  # TODO: differentiate between types of M1?
    ]
    base_stage_cat = CategoricalDtype(
        categories=['c - Clinical', 'p - Pathological', 'r - Reccurent'],
        ordered=True
    )
    df["אבחנה-Basic stage"] = df["אבחנה-Basic stage"].astype(base_stage_cat)
    base_stage_imputer = SimpleImputer(strategy="most_frequent")
    df["אבחנה-Basic stage"] = base_stage_imputer.fit_transform(
        df[["אבחנה-Basic stage"]]
    )
    df["אבחנה-Basic stage"] = df["אבחנה-Basic stage"].astype(base_stage_cat)

    hist_deg_cat = CategoricalDtype(
        categories=[
            'G1 - Well Differentiated',
            'G2 - Modereately well differentiated',
            'G3 - Poorly differentiated',
            'G4 - Undifferentiated'],
        ordered=True
    )
    df["אבחנה-Histopatological degree"] = df["אבחנה-Histopatological degree"].astype(hist_deg_cat)
    hist_deg_imputer = SimpleImputer(strategy="most_frequent")
    df["אבחנה-Histopatological degree"] = hist_deg_imputer.fit_transform(
        df[["אבחנה-Histopatological degree"]]
    )
    df["אבחנה-Histopatological degree"] = df["אבחנה-Histopatological degree"].astype(hist_deg_cat)

    lym_pen_cat = CategoricalDtype(
        categories=[
            'L0 - No Evidence of invasion',
            'LI - Evidence of invasion',
            'L1 - Evidence of invasion of superficial Lym.',
            'L2 - Evidence of invasion of depp Lym.'
            ], ordered=True
    )
    df["אבחנה-Lymphatic penetration"] = df["אבחנה-Lymphatic penetration"].astype(lym_pen_cat)
    hist_deg_imputer = SimpleImputer(
        strategy="constant",
        fill_value='L0 - No Evidence of invasion'  # TODO: fill based on other columns
    )
    df["אבחנה-Lymphatic penetration"] = hist_deg_imputer.fit_transform(
        df[["אבחנה-Lymphatic penetration"]]
    )
    df["אבחנה-Lymphatic penetration"] = df["אבחנה-Lymphatic penetration"].astype(lym_pen_cat)

    m_mark_cat = CategoricalDtype(
        categories=['M0', 'M1'],  # TODO: MX? NYE?
        ordered=True
    )

    df["אבחנה-M -metastases mark (TNM)"] = df["אבחנה-M -metastases mark (TNM)"].mask(
        ((df["אבחנה-M -metastases mark (TNM)"] == 'M1a') |
         (df["אבחנה-M -metastases mark (TNM)"] == 'M1b')), 'M1'
    ).astype(m_mark_cat)

    df["אבחנה-M -metastases mark (TNM)"] = df["אבחנה-M -metastases mark (TNM)"].astype(m_mark_cat)
    m_mark_imputer = SimpleImputer(strategy="most_frequent")
    df["אבחנה-M -metastases mark (TNM)"] = m_mark_imputer.fit_transform(
        df[["אבחנה-M -metastases mark (TNM)"]]
    )
    df["אבחנה-M -metastases mark (TNM)"] = df["אבחנה-M -metastases mark (TNM)"].astype(m_mark_cat)

    return [base_stage_imputer, hist_deg_imputer, hist_deg_imputer, m_mark_imputer]


def handle_side(df: pd.DataFrame):
    df["Side_right"] = (df["אבחנה-Side"] == 'ימין') | (df["אבחנה-Side"] == 'דו צדדי')
    df["Side_left"] = (df["אבחנה-Side"] == 'שמאל') | (df["אבחנה-Side"] == 'דו צדדי')

if __name__ == '__main__':
    args = docopt(__doc__)
    # print(args)
    if args["part1"]:
        train_X_fn = Path(args["--train-x"])
        train_y_fn = Path(args["--train-y"])
        # test_X_fn = Path(args["--test-x"])

        df = load_data(train_X_fn, train_y_fn)

        handle_numerical(df)
        df = handle_dates_features(df)
        df = handle_categorical_cols(df)
        preprocessing(df)

        handle_ordered_categories(df)
        handle_side(df)

        drop_cols(df, ['User Name',
                       'אבחנה-Her2',
                       'אבחנה-Ivi -Lymphovascular invasion',  # TODO
                       'אבחנה-KI67 protein',        # TODO
                       'אבחנה-N -lymph nodes mark (TNM)',  # TODO
                       'אבחנה-Side',
                       'אבחנה-Stage',    # TODO
                       'אבחנה-Surgery name1',   # TODO
                       'אבחנה-Surgery name2',   # TODO
                       'אבחנה-Surgery name3',  # TODO
                       'אבחנה-T -Tumor mark (TNM)',  # TODO
                       ])

        a = df.describe()
        # head = df.head(1000)
        # a = head.describe()
        # for colname, colval in df.iteritems():
        #     print(colname)
        #     print(pd.unique(head[colname]))
