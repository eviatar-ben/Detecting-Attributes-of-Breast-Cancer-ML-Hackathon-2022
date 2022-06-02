import pandas as pd
from sklearn.linear_model import LinearRegression
from explore_data import preprocessing
from main import handle_numerical, handle_ordered_categories, handle_side
from preprocessor import handle_categorical_cols, drop_cols

def load_data(train_X_fn):
    features = pd.read_csv(train_X_fn, parse_dates=[
        "אבחנה-Diagnosis date",
        "אבחנה-Surgery date1",
        "אבחנה-Surgery date2",
        "אבחנה-Surgery date3",
        "surgery before or after-Activity date"
    ], infer_datetime_format=True, dayfirst=True)
    return features


if __name__ == '__main__':
    df = load_data("splited_datasets/X_train_base1")
    labels = pd.read_csv("splited_datasets/y_train_base1")

    handle_numerical(df)

    df = handle_categorical_cols(df)
    preprocessing(df)

    handle_ordered_categories(df)
    handle_side(df)

    drop_cols(df, ['User Name',
                   'אבחנה-Her2',
                   'אבחנה-Ivi -Lymphovascular invasion',  # TODO
                   'אבחנה-KI67 protein',  # TODO
                   'אבחנה-N -lymph nodes mark (TNM)',  # TODO
                   'אבחנה-Side',
                   'אבחנה-Stage',  # TODO
                   'אבחנה-Surgery name1',  # TODO
                   'אבחנה-Surgery name2',  # TODO
                   'אבחנה-Surgery name3',  # TODO
                   'אבחנה-T -Tumor mark (TNM)',  # TODO
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

    print(df)
    base_model = LinearRegression()
    base_model.fit(df, labels)
