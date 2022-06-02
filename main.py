""" Usage:
    <file-name> (part1 | part2 | part3) --train-x=TRAIN_X --train-y=TRAIN_Y [--test-x=TEST_X]
    <file-name> (part1 | part2 | part3) --train-x=TRAIN_X --train-y=TRAIN_Y [--test-x=TEST_X --test-y=TEST_ --out=PRED]

Options:
  --help                           Show this message and exit
"""
from pandas import CategoricalDtype     # TODO: pd.CategoricalDtype instead
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from docopt import docopt
from pathlib import Path
import logging
import pandas as pd
from typing import Tuple, Iterable
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, MultiLabelBinarizer
from preprocessor import *
from explore_data import *
import plotly.graph_objects as go
import plotly.express as px


def load_data(train_X_fn: Path, train_y_fn: Path):
    features = pd.read_csv(train_X_fn, parse_dates=[
        "אבחנה-Diagnosis date",
        "אבחנה-Surgery date1",
        "אבחנה-Surgery date2",
        "אבחנה-Surgery date3",
        "surgery before or after-Activity date"
    ], infer_datetime_format=True, dayfirst=True)

    labels = pd.read_csv(train_y_fn)
    labels["אבחנה-Location of distal metastases"] = labels["אבחנה-Location of distal metastases"].apply(eval)
    full_data = features
    full_data["אבחנה-Location of distal metastases"] = labels["אבחנה-Location of distal metastases"]
    full_data = full_data.loc[:, ~full_data.columns.str.contains('^Unnamed')]
    full_data.reset_index(inplace=True, drop=True)
    return full_data


def handle_numerical(df: pd.DataFrame, imputers=None) -> Iterable[SimpleImputer]:
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
    if imputers is not None:
        median_imputer = imputers[0]

    df["אבחנה-Age"] = median_imputer.fit_transform(df[["אבחנה-Age"]])
    # df["אבחנה-Age"] = median_imputer.transform(df["אבחנה-Age"])
    return [median_imputer]


def handle_ordered_categories(df: pd.DataFrame, imputers=None) -> Iterable[SimpleImputer]:
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
    if imputers is not None:
        base_stage_imputer = imputers[0]

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
    if imputers is not None:
        hist_deg_imputer = imputers[1]

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
    if imputers is not None:
        hist_deg_imputer = imputers[2]

    df["אבחנה-Lymphatic penetration"] = hist_deg_imputer.fit_transform(
        df[["אבחנה-Lymphatic penetration"]]
    )
    df["אבחנה-Lymphatic penetration"] = df["אבחנה-Lymphatic penetration"].astype(lym_pen_cat)

    # m_mark_cat = CategoricalDtype(
    #     categories=['M0', 'M1'],  # TODO: MX? NYE?
    #     ordered=True
    # )
    #
    # df["אבחנה-M -metastases mark (TNM)"] = df["אבחנה-M -metastases mark (TNM)"].mask(
    #     ((df["אבחנה-M -metastases mark (TNM)"] == 'M1a') |
    #      (df["אבחנה-M -metastases mark (TNM)"] == 'M1b')), 'M1'
    # ).astype(m_mark_cat)
    #
    # df["אבחנה-M -metastases mark (TNM)"] = df["אבחנה-M -metastases mark (TNM)"].astype(m_mark_cat)
    # m_mark_imputer = SimpleImputer(strategy="most_frequent")
    # df["אבחנה-M -metastases mark (TNM)"] = m_mark_imputer.fit_transform(
    #     df[["אבחנה-M -metastases mark (TNM)"]]
    # )
    # df["אבחנה-M -metastases mark (TNM)"] = df["אבחנה-M -metastases mark (TNM)"].astype(m_mark_cat)

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    return [base_stage_imputer, hist_deg_imputer, hist_deg_imputer]


def handle_side(df: pd.DataFrame):
    df["Side_right"] = (df["אבחנה-Side"] == 'ימין') | (df["אבחנה-Side"] == 'דו צדדי')
    df["Side_left"] = (df["אבחנה-Side"] == 'שמאל') | (df["אבחנה-Side"] == 'דו צדדי')


if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    if args["part1"]:
        train_X_fn = Path(args["--train-x"])
        train_y_fn = Path(args["--train-y"])

        df = load_data(train_X_fn, train_y_fn)

        num_imp = handle_numerical(df)
        # df = handle_dates_features(df)
        df, encoder = handle_categorical_cols(df)
        # df = handle_ki67(df)
        df = handle_ivi(df)
        preprocessing(df)

        ord_imp = handle_ordered_categories(df)
        handle_side(df)

        drop_cols(df, ['User Name',
                       'אבחנה-Her2',
                       # 'אבחנה-Ivi -Lymphovascular invasion',
                       'אבחנה-KI67 protein',        # TODO
                       'אבחנה-N -lymph nodes mark (TNM)',
                       'אבחנה-Side',
                       'אבחנה-Stage',
                       'אבחנה-Surgery name1',   # TODO
                       'אבחנה-Surgery name2',   # TODO
                       'אבחנה-Surgery name3',  # TODO
                       'אבחנה-T -Tumor mark (TNM)',
                       "אבחנה-M -metastases mark (TNM)",
                       'אבחנה-Tumor depth',     # TODO
                       'אבחנה-Tumor width',     # TODO
                       'אבחנה-er',
                       'אבחנה-pr',
                       'id-hushed_internalpatientid',
                       'surgery before or after-Actual activity',   # TODO
                       # TODO: retry dates with manual parse for Unknowns?
                       'אבחנה-Surgery date1',
                       'אבחנה-Surgery date2',
                       'אבחנה-Surgery date3',
                       'surgery before or after-Activity date',
                       'אבחנה-Diagnosis date',
                       ])

        mlb = MultiLabelBinarizer()
        transformed_y = mlb.fit_transform(df["אבחנה-Location of distal metastases"])
        transformed_y_df = pd.DataFrame(transformed_y, columns=mlb.classes_)
        result = pd.concat([df, transformed_y_df], axis=1).drop(
            ["אבחנה-Location of distal metastases"], axis=1)

        a = result.describe()

        if args['--test-y'] is not None:
            baseline = RandomForestClassifier()
            baseline.fit(df.drop(["אבחנה-Location of distal metastases"], axis=1), transformed_y_df)

            train_X_fn = Path(args["--test-x"])
            train_y_fn = Path(args["--test-y"])

            df = load_data(train_X_fn, train_y_fn)

            handle_numerical(df, num_imp)
            # df = handle_dates_features(df)
            df, encoder = handle_categorical_cols(df, encoder)
            # df = handle_ki67(df)
            df = handle_ivi(df)
            preprocessing(df)

            handle_ordered_categories(df, ord_imp)
            handle_side(df)

            drop_cols(df, ['User Name',
                           'אבחנה-Her2',
                           # 'אבחנה-Ivi -Lymphovascular invasion',
                           'אבחנה-KI67 protein',  # TODO
                           'אבחנה-N -lymph nodes mark (TNM)',
                           'אבחנה-Side',
                           'אבחנה-Stage',
                           'אבחנה-Surgery name1',  # TODO
                           'אבחנה-Surgery name2',  # TODO
                           'אבחנה-Surgery name3',  # TODO
                           'אבחנה-T -Tumor mark (TNM)',
                           "אבחנה-M -metastases mark (TNM)",
                           'אבחנה-Tumor depth',  # TODO
                           'אבחנה-Tumor width',  # TODO
                           'אבחנה-er',
                           'אבחנה-pr',
                           'id-hushed_internalpatientid',
                           'surgery before or after-Actual activity',  # TODO
                           # TODO: retry dates with manual parse for Unknowns?
                           'אבחנה-Surgery date1',
                           'אבחנה-Surgery date2',
                           'אבחנה-Surgery date3',
                           'surgery before or after-Activity date',
                           'אבחנה-Diagnosis date',
                           ])
            transformed_y = mlb.transform(
                df["אבחנה-Location of distal metastases"])
            transformed_y_df = pd.DataFrame(transformed_y,
                                            columns=mlb.classes_)

            pred = baseline.predict(df.drop(["אבחנה-Location of distal metastases"], axis=1))
            mcm = multilabel_confusion_matrix(transformed_y_df, pred)

            out_path = Path(args["--out"])
            combined = pd.DataFrame({"אבחנה-Location of distal metastases": mlb.inverse_transform(pred)})
            combined.to_csv(path_or_buf=out_path, index=False)



        # PCA::::
        # pca = PCA(n_components=2)
        # tran_pca = pca.fit_transform(df.drop(["אבחנה-Location of distal metastases"], axis=1))
        # fig = px.scatter(x=tran_pca[:, 0], y=tran_pca[:, 1], color=(transformed_y_df.any(axis=1)))
        # fig.show()

