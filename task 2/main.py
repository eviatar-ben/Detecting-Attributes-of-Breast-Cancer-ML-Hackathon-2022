""" Usage:
    <file-name> (part1 | part2) pred --train-x=TRAIN_X --train-y=TRAIN_Y [--parsed=OUT_FILE] --test-x=TEST_X --out=PRED [options]
    <file-name> (part1 | part2) --cv=K --train-x=TRAIN_X --train-y=TRAIN_Y [--parsed=OUT_FILE] [options]
    <file-name> (part1 | part2) test --train-x=TRAIN_X --train-y=TRAIN_Y --test-x=TEST_X --test-y=TEST_Y --out=PRED [--parsed=OUT_FILE] [options]
    <file-name> (part1 | part2) baseline --train-x=TRAIN_X --train-y=TRAIN_Y --test-x=TEST_X --test-y=TEST_Y --out=PRED [--parsed=OUT_FILE] [options]
    <file-name> part3 --train-x=TRAIN_X [options]

Options:
  --help            # Show this message and exit
  --seed=SEED       [default: 0]
"""

from docopt import docopt
from sklearn.cluster import KMeans
from sklearn.covariance import empirical_covariance
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.model_selection import cross_validate, KFold
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px

from explore_data import *
from preprocessor import *


def part_1(args):
    # Parse train:
    train_X_fn = Path(args["--train-x"])
    train_y_fn = Path(args["--train-y"])
    df = load_data_part_1(train_X_fn, train_y_fn)

    df, num_imp, ord_imp, encoder = parse_features(df)
    features = df.drop(["אבחנה-Location of distal metastases"], axis=1).drop_duplicates()
    df = df.loc[features.index]

    # Save parsed data
    if args['--parsed'] is not None:
        parsed_fn = Path(args['--parsed'])
        df.to_csv(parsed_fn, index=False)

    mlb = MultiLabelBinarizer(classes=[
        'PUL - Pulmonary',
        'BON - Bones',
        'SKI - Skin',
        'LYM - Lymph nodes',
        'BRA - Brain',
        'HEP - Hepatic',
        'PER - Peritoneum',
        'PLE - Pleura',
        'OTH - Other',
        'ADR - Adrenals',
        'MAR - Bone Marrow',
    ])
    transformed_y = mlb.fit_transform(
        df["אבחנה-Location of distal metastases"])
    transformed_y_df = pd.DataFrame(transformed_y, columns=mlb.classes_)

    model = ClassifierChain(DecisionTreeClassifier())

    # Make prediction:
    if args['pred']:
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
            {"אבחנה-Location of distal metastases":
                 mlb.inverse_transform(pred)}
        )
        combined.to_csv(path_or_buf=out_path, index=False)

    # Evaluate test:
    if args['baseline'] or args['test']:
        if args['baseline']:
            baseline = DecisionTreeClassifier(max_depth=2)
            baseline.fit(
                df.drop(["אבחנה-Location of distal metastases"], axis=1),
                transformed_y_df)
            model = baseline
        else:
            model.fit(df.drop(["אבחנה-Location of distal metastases"], axis=1).astype(float),
                      transformed_y_df)

        train_X_fn = Path(args["--test-x"])
        train_y_fn = Path(args["--test-y"])

        df = load_data_part_1(train_X_fn, train_y_fn)

        df, num_imp, ord_imp, encoder = parse_features(df, num_imp, ord_imp,
                                                       encoder)

        features = df.drop(["אבחנה-Location of distal metastases"],
                           axis=1).drop_duplicates()
        df = df.loc[features.index]

        pred = model.predict(
            df.drop(["אבחנה-Location of distal metastases"], axis=1).astype(float))

        transformed_y = mlb.transform(
            df["אבחנה-Location of distal metastases"])
        transformed_y_df = pd.DataFrame(transformed_y, columns=mlb.classes_)

        mcm = multilabel_confusion_matrix(transformed_y_df, pred)
        print(mcm)
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
            ClassifierChain(DecisionTreeClassifier()),
            ClassifierChain(ExtraTreesClassifier()),
            ClassifierChain(RandomForestClassifier()),
            ClassifierChain(RandomForestClassifier(class_weight="balanced_subsample")),
            ClassifierChain(DecisionTreeClassifier(class_weight="balanced")),
            RandomForestClassifier(class_weight="balanced"),
            RandomForestClassifier(class_weight="balanced_subsample"),
            RandomForestClassifier(),
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
            print("## f1_micro Test ##")
            print(np.mean(scores["test_f1_micro"]))
            print(scores["test_f1_micro"])
            print("## f1_micro Train ##")
            print(np.mean(scores["train_f1_micro"]))


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
    features = df.drop_duplicates()
    df = df.loc[features.index]
    labels = labels.loc[df.index]

    # Save trained model:
    if args['--parsed'] is not None:
        df['אבחנה-Tumor size'] = labels['אבחנה-Tumor size']
        parsed_fn = Path(args['--parsed'])
        df.to_csv(parsed_fn, index=False)
        df.drop(['אבחנה-Tumor size'], axis=1, inplace=True)

    model = RandomForestRegressor(max_features=0.75, max_samples=0.75)
    # Make prediction:
    if args['pred']:
        model.fit(df, labels.to_numpy().T[0])

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
        pred = np.maximum(pred, 0)
        out_path = Path(args["--out"])
        combined = pd.DataFrame(pred, columns=['אבחנה-Tumor size'])
        combined.to_csv(path_or_buf=out_path, index=False)

    # Test:
    if args['baseline'] or args['test']:
        if args['baseline']:
            model = LinearRegression()
            model.fit(df, labels.to_numpy().T[0])
        else:
            model.fit(df, labels.to_numpy().T[0])

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
        features = df.drop_duplicates()
        df = df.loc[features.index]
        labels = labels.loc[df.index]

        pred = model.predict(df)

        pred = np.maximum(pred, 0)

        mcm = confusion_matrix(labels.to_numpy().T[0], pred)
        print(mcm)

        out_path = Path(args["--out"])
        combined = pd.DataFrame(pred, columns=['אבחנה-Tumor size'])
        combined.to_csv(path_or_buf=out_path, index=False)

    # Test using cross validation
    if args["--cv"] is not None:
        splits = int(args["--cv"])
        scores = cross_validate(model, df, labels.to_numpy().T[0], cv=splits,
                                scoring='neg_mean_squared_error',
                                return_train_score=True,
                                return_estimator=True)
        print(model)
        print("neg MSE test:")
        print(np.mean(scores["test_score"]))
        print(scores["test_score"])
        print("neg MSE train:")
        print(np.mean(scores["train_score"]))


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
    fig = px.scatter_3d(x=tran_pca[:, 0], y=tran_pca[:, 1], z=tran_pca[:, 2],
                        color=df['stage processed'], title="PCA components analysis over cancer stage")
    fig.show()

    cluster = KMeans(n_clusters=3)
    feat = cluster.fit_transform(df)
    px.scatter_3d(x=feat[:, 0], y=feat[:, 1], z=feat[:, 2], title="KMeans clustering with age as color",
                  color=df['אבחנה-Age']).show()

    fig = px.imshow(empirical_covariance(df.astype(float)), title="Feature covariance")
    fig.show()



# part1 pred --train-x=train.feats.csv --train-y=train.labels.0.csv --test-x=test.feats.csv --out=./part1/predictions.csv
# part1 --cv=5 --train-x=train.feats.csv --train-y=train.labels.0.csv --seed=800835
# part2 pred --train-x=train.feats.csv --train-y=train.labels.1.csv --test-x=test.feats.csv --out=./part2/predictions.csv
# part2 --cv=8 --train-x=train.feats.csv --train-y=train.labels.1.csv --seed=800835
# part3 --train-x=train.feats.csv
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
