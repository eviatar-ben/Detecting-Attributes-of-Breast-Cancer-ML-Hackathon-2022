""" Usage:
    <file-name> (part1 | part2) pred --train-x=TRAIN_X --train-y=TRAIN_Y [--parsed=OUT_FILE] --test-x=TEST_X --out=PRED [options]
    <file-name> (part1 | part2) --cv=K --train-x=TRAIN_X --train-y=TRAIN_Y [--parsed=OUT_FILE] [options]
    <file-name> (part1 | part2) test --train-x=TRAIN_X --train-y=TRAIN_Y --test-x=TEST_X --test-y=TEST_Y --out=PRED [--parsed=OUT_FILE] [options]
    <file-name> (part1 | part2) baseline --train-x=TRAIN_X --train-y=TRAIN_Y --test-x=TEST_X --test-y=TEST_Y --out=PRED [--parsed=OUT_FILE] [options]
    <file-name> part3 --train-x=TRAIN_X [options]

Options:
  --help                           Show this message and exit
  --seed=SEED                       [default: 0]
"""
import tqdm
from pandas import CategoricalDtype  # TODO: pd.CategoricalDtype instead
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    BaggingClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier, \
    RidgeClassifierCV
from sklearn.impute import SimpleImputer
from docopt import docopt
from pathlib import Path
import logging
import pandas as pd
from typing import Tuple, Iterable
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder, MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.problem_transform import LabelPowerset

from MultiLabelClassifier import build_model
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

    # df['אבחנה-Surgery sum'].mask(
    #     (df['אבחנה-Surgery sum'].isna()),
    #     (df['אבחנה-Surgery date1'].notna()) +
    #     (df['אבחנה-Surgery date2'].notna()) +
    #     (df['אבחנה-Surgery date3'].notna()),
    #     inplace=True
    # )
    df['אבחנה-Surgery sum'] = df['אבחנה-Surgery sum'].fillna(0).astype(float)

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
    df["Basic stage"] = df["אבחנה-Basic stage"].astype(base_stage_cat)
    df["Basic stage"] = df["Basic stage"].cat.codes.fillna(-10)

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
    df["Histopatological degree"] = df["אבחנה-Histopatological degree"].astype(hist_deg_cat)
    df["Histopatological degree"] = df["Histopatological degree"].cat.codes.fillna(-10)

    lym_pen_cat = CategoricalDtype(
        categories=[
            'L0 - No Evidence of invasion',
            'LI - Evidence of invasion',
            'L1 - Evidence of invasion of superficial Lym.',
            'L2 - Evidence of invasion of depp Lym.'
        ], ordered=True
    )
    df["אבחנה-Lymphatic penetration"] = df["אבחנה-Lymphatic penetration"].astype(lym_pen_cat)
    lym_pen_imputer = SimpleImputer(
        strategy="constant",
        fill_value=np.nan  # TODO: fill based on other columns
    )
    if imputers is not None:
        lym_pen_imputer = imputers[2]

    df["אבחנה-Lymphatic penetration"] = lym_pen_imputer.fit_transform(
        df[["אבחנה-Lymphatic penetration"]]
    )
    df["Lymphatic penetration"] = df["אבחנה-Lymphatic penetration"].astype(lym_pen_cat)
    df["Lymphatic penetration"] = df["Lymphatic penetration"].cat.codes.fillna(-10)

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    return [base_stage_imputer, lym_pen_imputer, hist_deg_imputer]


def handle_side(df: pd.DataFrame):
    df["Side_right"] = (df["אבחנה-Side"] == 'ימין') | (df["אבחנה-Side"] == 'דו צדדי')
    df["Side_left"] = (df["אבחנה-Side"] == 'שמאל') | (df["אבחנה-Side"] == 'דו צדדי')


def parse_features(df: pd.DataFrame, num_imp=None, ord_imp=None, encoder=None):
    num_imp = handle_numerical(df, num_imp)
    # df = handle_dates_features(df)
    df, encoder = handle_categorical_cols(df, encoder)
    df = handle_ivi(df)
    df = handle_ki67(df)
    preprocessing(df)
    ord_imp = handle_ordered_categories(df, ord_imp)

    handle_side(df)
    drop_cols(df, ['אבחנה-Histological diagnosis',
                   # 'אבחנה-Ivi -Lymphovascular invasion',
                   # 'אבחנה-KI67 protein',
                   # 'User Name',
                   'אבחנה-Her2',
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
                   # ' Form Name',
                   "אבחנה-Basic stage",
                   "אבחנה-Histopatological degree",
                   "אבחנה-Lymphatic penetration",
                   ' Hospital',
                   'אבחנה-Margin Type'
                   ])

    # df = df[["אבחנה-Location of distal metastases"]]
    return df, num_imp, ord_imp, encoder


def multi():
    import MultiLabelClassifier
    # X_train = np.array(pd.DataFrame.to_numpy(X_train), dtype=float)
    return MultiLabelClassifier.get_models()


def part_1(args):
    # Parse train:
    train_X_fn = Path(args["--train-x"])
    train_y_fn = Path(args["--train-y"])
    df = load_data(train_X_fn, train_y_fn)

    df, num_imp, ord_imp, encoder = parse_features(df)
    features = df.drop(["אבחנה-Location of distal metastases"], axis=1).drop_duplicates()
    df = df.loc[features.index]

    # Save parsed data
    if args['--parsed'] is not None:
        parsed_fn = Path(args['--parsed'])
        df.to_csv(parsed_fn, index=False)

    mlb = MultiLabelBinarizer()
    transformed_y = mlb.fit_transform(
        df["אבחנה-Location of distal metastases"])
    transformed_y_df = pd.DataFrame(transformed_y, columns=mlb.classes_)
    # df = pd.concat([df, transformed_y_df], axis=1)
    # Make prediction:
    if args['pred']:
        model = LabelPowerset(RandomForestClassifier(ccp_alpha=0.00001))
        model.fit(df.drop(["אבחנה-Location of distal metastases"], axis=1).astype(float),
                  transformed_y_df)

        train_X_fn = Path(args["--test-x"])
        features = pd.read_csv(train_X_fn, parse_dates=[
            "אבחנה-Diagnosis date",
            "אבחנה-Surgery date1",
            "אבחנה-Surgery date2",
            "אבחנה-Surgery date3",
            "surgery before or after-Activity date"
        ], infer_datetime_format=True, dayfirst=True)

        features, num_imp, ord_imp, encoder = parse_features(features, num_imp,
                                                             ord_imp, encoder)
        pred = model.predict(features)
        out_path = Path(args["--out"])
        combined = pd.DataFrame(
            {"אבחנה-Location of distal metastases": mlb.inverse_transform(
                pred)}
        )
        combined.to_csv(path_or_buf=out_path, index=False)

    # Evaluate test:
    if args['baseline'] or args['test']:
        model = None
        if args['baseline']:
            baseline = DecisionTreeClassifier(max_depth=2)
            baseline.fit(
                df.drop(["אבחנה-Location of distal metastases"], axis=1),
                transformed_y_df)
            model = baseline
        else:
            model = LabelPowerset(DecisionTreeClassifier())
            model.fit(df.drop(["אבחנה-Location of distal metastases"], axis=1).astype(float),
                      transformed_y_df)

        train_X_fn = Path(args["--test-x"])
        train_y_fn = Path(args["--test-y"])

        df = load_data(train_X_fn, train_y_fn)

        df, num_imp, ord_imp, encoder = parse_features(df, num_imp, ord_imp,
                                                       encoder)

        features = df.drop(["אבחנה-Location of distal metastases"],
                           axis=1).drop_duplicates()
        df = df.loc[features.index]

        pred = model.predict(
            df.drop(["אבחנה-Location of distal metastases"], axis=1))

        transformed_y = mlb.transform(
            df["אבחנה-Location of distal metastases"])
        transformed_y_df = pd.DataFrame(transformed_y, columns=mlb.classes_)

        mcm = multilabel_confusion_matrix(transformed_y_df, pred)

        out_path = Path(args["--out"])
        combined = pd.DataFrame({
            "אבחנה-Location of distal metastases": mlb.inverse_transform(
                pred)})
        combined.to_csv(path_or_buf=out_path, index=False)

    # Evaluate cross validation:
    if args["--cv"] is not None:
        features = df.drop(["אבחנה-Location of distal metastases"], axis=1).astype(float)
        labels = transformed_y_df
        splits = int(args["--cv"])

        models = [
                DecisionTreeClassifier(max_depth=18),
                RandomForestClassifier(),
                  RidgeClassifierCV(),
                  LabelPowerset(DecisionTreeClassifier()),
                  LabelPowerset(RandomForestClassifier()),
                  ]
        # models += [i for i in multi()]
        for model in models:
            scores = cross_validate(model, features, labels, cv=KFold(n_splits=splits, shuffle=True),
                                    scoring=['f1_micro', 'f1_macro'],
                                    return_train_score=True,
                                    return_estimator=True)
            print(model)
            print("## f1_macro Test ##")
            print(np.mean(scores["test_f1_macro"]))
            print(scores["test_f1_macro"])
            print("## f1_macro Train ##")
            print(np.mean(scores["train_f1_macro"]))
            print(scores["train_f1_macro"])
            print("## f1_micro Test ##")
            print(np.mean(scores["test_f1_micro"]))
            print(scores["test_f1_micro"])
            print("## f1_micro Train ##")
            print(np.mean(scores["train_f1_micro"]))
            print(scores["train_f1_micro"])


def part_2(args):
    # Parse train:
    train_X_fn = Path(args["--train-x"])
    train_y_fn = Path(args["--train-y"])
    labels = pd.read_csv(train_y_fn)

    df = pd.read_csv(train_X_fn, parse_dates=[
        "אבחנה-Diagnosis date",
        "אבחנה-Surgery date1",
        "אבחנה-Surgery date2",
        "אבחנה-Surgery date3",
        "surgery before or after-Activity date"
    ], infer_datetime_format=True, dayfirst=True)

    df, num_imp, ord_imp, encoder = parse_features(df)

    # Save trained model:  TODO: load trained model - requires saving the dtypes?
    if args['--parsed'] is not None:
        df['אבחנה-Tumor size'] = labels['אבחנה-Tumor size']
        parsed_fn = Path(args['--parsed'])
        df.to_csv(parsed_fn, index=False)
        df.drop(['אבחנה-Tumor size'], axis=1, inplace=True)

    # Make prediction:
    if args['pred']:
        model = LinearRegression()
        model.fit(df, labels)

        train_X_fn = Path(args["--test-x"])
        features = pd.read_csv(train_X_fn, parse_dates=[
            "אבחנה-Diagnosis date",
            "אבחנה-Surgery date1",
            "אבחנה-Surgery date2",
            "אבחנה-Surgery date3",
            "surgery before or after-Activity date"
        ], infer_datetime_format=True, dayfirst=True)

        features, num_imp, ord_imp, encoder = parse_features(features, num_imp,
                                                             ord_imp, encoder)
        pred = model.predict(features)
        out_path = Path(args["--out"])
        combined = pd.DataFrame(pred, columns=['אבחנה-Tumor size'])
        combined.to_csv(path_or_buf=out_path, index=False)

    # Test:
    if args['baseline'] or args['test']:
        model = None
        if args['baseline']:
            model = LinearRegression()
            model.fit(df, labels)
        else:
            model = LinearRegression()
            model.fit(df, labels)

        train_X_fn = Path(args["--test-x"])
        train_y_fn = Path(args["--test-y"])

        labels = pd.read_csv(train_y_fn)

        df = pd.read_csv(train_X_fn, parse_dates=[
            "אבחנה-Diagnosis date",
            "אבחנה-Surgery date1",
            "אבחנה-Surgery date2",
            "אבחנה-Surgery date3",
            "surgery before or after-Activity date"
        ], infer_datetime_format=True, dayfirst=True)

        df, num_imp, ord_imp, encoder = parse_features(df, num_imp, ord_imp,
                                                       encoder)
        pred = model.predict(df)

        mcm = confusion_matrix(labels, pred)

        out_path = Path(args["--out"])
        combined = pd.DataFrame(pred, columns=['אבחנה-Tumor size'])
        combined.to_csv(path_or_buf=out_path, index=False)

    # Test using cross validation
    if args["--cv"] is not None:
        splits = int(args["--cv"])
        model = LinearRegression()
        scores = cross_validate(model, df, labels, cv=splits,
                                scoring=['f1_micro', 'f1_macro'],
                                return_train_score=True,
                                return_estimator=True)
        print("## f1_macro ##")
        print(np.mean(scores["test_f1_macro"]))
        print(scores["test_f1_macro"])
        print(np.mean(scores["train_f1_macro"]))
        print(scores["train_f1_macro"])
        print("## f1_micro ##")
        print(np.mean(scores["test_f1_micro"]))
        print(scores["test_f1_micro"])
        print(np.mean(scores["rain_f1_micro"]))
        print(scores["train_f1_micro"])


def part_3(args):
    train_X_fn = Path(args["--train-x"])
    df = pd.read_csv(train_X_fn, parse_dates=[
        "אבחנה-Diagnosis date",
        "אבחנה-Surgery date1",
        "אבחנה-Surgery date2",
        "אבחנה-Surgery date3",
        "surgery before or after-Activity date"
    ], infer_datetime_format=True, dayfirst=True)

    df, num_imp, ord_imp, encoder = parse_features(df)
    # PCA::::
    pca = PCA(n_components=3)
    tran_pca = pca.fit_transform(df.astype(float))
    fig = px.scatter_3d(x=tran_pca[:, 0], y=tran_pca[:, 1], z=tran_pca[:, 2], color=df['stage processed'])
    fig.show()


# part1 pred --train-x=splited_datasets/X_train.csv --train-y=splited_datasets/y_train.csv --test-x=splited_datasets/X_test.csv --out=./new_val_pred.csv
# part1 --cv=5 --train-x=splited_datasets/X_train_more.csv --train-y=splited_datasets/y_train_more.csv
# part3 --train-x=splited_datasets/X_train_more.csv
if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    seed = 0
    if args['--seed'] is not None:
        seed = int(args['--seed'])
    np.random.seed(seed)
    if args["part1"]:
        part_1(args)
    if args["part2"]:
        part_2(args)
    if args["part3"]:
        part_3(args)
