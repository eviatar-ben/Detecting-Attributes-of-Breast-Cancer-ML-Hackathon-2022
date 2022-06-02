import pandas as pd
import numpy as np
import utilities


def handle_ordered_categorical_cols(df):
    # 'Histological Diagnosis'
    pass


def handle_categorical_cols(df, encoder=None):
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    # 'Form Name'
    categorical_cols = [' Form Name', ' Hospital', 'אבחנה-Histological diagnosis',
                        'אבחנה-Margin Type']  # TODO 'אבחנה-Basic stage',

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        encoder.fit(df[categorical_cols])
    transformed = encoder.transform(df[categorical_cols])
    transformed_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out(categorical_cols))
    result = pd.concat([df, transformed_df], axis=1)
    return result.drop(categorical_cols, axis=1), encoder


def drop_cols(df, cols):
    # User name
    for col_name in cols:
        df.drop(col_name, axis=1, inplace=True)


def handle_dates_features(df):
    # handle unknown dates in order to subtract dates:
    # in total _ samples were dropped:
    # (df['אבחנה-Surgery date1'] == 'Unknown').sum()
    # (df['אבחנה-Surgery date2'] == 'Unknown').sum()
    # (df['אבחנה-Surgery date3'] == 'Unknown').sum()

    df = df[df['אבחנה-Surgery date1'] != 'Unknown']
    df = df[df['אבחנה-Surgery date2'] != 'Unknown']
    df = df[df['אבחנה-Surgery date3'] != 'Unknown']
    df.reset_index()

    # 'diagnosis_and_surgery_days_dif'  # 33-7
    dif = pd.to_datetime(df['אבחנה-Diagnosis date']) - pd.to_datetime(df['surgery before or after-Activity date'])
    df['diagnosis_and_surgery_days_dif'] = dif.dt.days

    # 'first_and_second_surgery_days_diff' # 22-21
    dif = pd.to_datetime(df['אבחנה-Surgery date2']) - pd.to_datetime(df['אבחנה-Surgery date1'])
    df['first_and_second_surgery_days_diff'] = dif.dt.days

    # 'second_and_third_surgery_days_diff' # 23-22
    dif = pd.to_datetime(df['אבחנה-Surgery date3']) - pd.to_datetime(df['אבחנה-Surgery date2'])
    df['second_and_third_surgery_days_diff'] = dif.dt.days

    # drop
    drop_cols(df, ['אבחנה-Surgery date1', 'אבחנה-Surgery date2', 'אבחנה-Surgery date3',
                   'surgery before or after-Activity date', 'אבחנה-Diagnosis date'])
    return df


def handle_ivi(df):
    # utilities.present_unique_values(df, col_name='אבחנה-Ivi -Lymphovascular invasion')
    positive_val = ['yes', '+', 'extensive', 'pos', 'MICROPAPILLARY VARIANT', '(+)']
    negative_val = ['not', 'none', 'neg', 'no', '-', '(-)', 'NO', 'No']

    last = df['אבחנה-Ivi -Lymphovascular invasion'] == 'yes'

    for pos_val in positive_val:
        cur = df['אבחנה-Ivi -Lymphovascular invasion'] == pos_val
        last = cur | last
    df['pos_ivi'] = last

    last = df['אבחנה-Ivi -Lymphovascular invasion'] == 'not'
    for neg_val in negative_val:
        cur = df['אבחנה-Ivi -Lymphovascular invasion'] == neg_val
        last = cur | last
    df['neg_ivi'] = last

    drop_cols(df, ['אבחנה-Ivi -Lymphovascular invasion'])
    return df


def handle_ki67(df):
    def get_low():
        words = ['Score 1', 'Score1-2', 'Very Low <3%', 'low-int']
        result = []
        for val in unique_vals:
            for i in range(1, 20):
                if str(i) in val:
                    result.append(val)
        unique_values_minus_result = [val for val in unique_vals if val not in result]
        return result, unique_values_minus_result

    def get_medium():
        words = ['score1-2', 'score 2', 'Score 2', 'Score II']
        result = []
        for val in unique_vals:
            for i in range(20, 50):
                if str(i) in val:
                    result.append(val)
        unique_values_minus_result = [val for val in unique_vals if val not in result]
        return result, unique_values_minus_result

    def get_medium_high():
        result = []
        for val in unique_vals:
            for i in range(50, 70):
                if str(i) in val:
                    result.append(val)
        unique_values_minus_result = [val for val in unique_vals if val not in result]
        return result, unique_values_minus_result

    def get_high():
        words = ['score 3-4', 'score 3', 'High', 'Score 4', 'high', 'HIGH']
        result = []
        for val in unique_vals:
            for i in range(70, 100):
                if str(i) in val:
                    result.append(val)
        unique_values_minus_result = [val for val in unique_vals if val not in result]
        return result, unique_values_minus_result

    # utilities.present_unique_values(df, 'אבחנה-KI67 protein')
    unique_vals = df['אבחנה-KI67 protein'].unique()[1:]
    high, unique_vals = get_high()
    medium_high, unique_vals = get_medium_high()
    medium, unique_vals = get_medium()
    low, unique_vals = get_low()

    low_indices = set()
    for val in low:
        cur = df['אבחנה-KI67 protein'] == val
        low_indices |= set(cur[cur].index)
    df.loc[low_indices, 'אבחנה-KI67 protein'] = 20

    medium_indices = set()
    for val in medium:
        cur = df['אבחנה-KI67 protein'] == val
        medium_indices |= set(cur[cur].index)
    df.loc[medium_indices, 'אבחנה-KI67 protein'] = 40

    medium_high_indices = set()
    for val in medium_high:
        cur = df['אבחנה-KI67 protein'] == val
        medium_high_indices |= set(cur[cur].index)
    df.loc[medium_high_indices, 'אבחנה-KI67 protein'] = 60

    high_indices = set()
    for val in high:
        cur = df['אבחנה-KI67 protein'] == val
        high_indices |= set(cur[cur].index)
    df.loc[high_indices, 'אבחנה-KI67 protein'] = 80

    # todo: decide based on correlation nan values
    nan_idx = df[df['אבחנה-KI67 protein'].isnull()].index.tolist()
    df['אבחנה-KI67 protein'][nan_idx] = 10

    # todo: handle with values that not appears in the list ahead
    union_idx = {*low_indices, *medium_indices, *medium_high_indices, *high_indices, *nan_idx}
    different_values = [i for i in range(len(df['אבחנה-KI67 protein'])) if i not in union_idx]

    df['אבחנה-KI67 protein'][different_values] = 20

    return df


def main():
    df = pd.read_csv(r'splited_datasets/features_train_base_0.csv', parse_dates=[
        "אבחנה-Diagnosis date",
        "אבחנה-Surgery date1",
        "אבחנה-Surgery date2",
        "אבחנה-Surgery date3",
        "surgery before or after-Activity date"
    ], infer_datetime_format=True, dayfirst=True)
    # drop_cols(df, ['User Name'])
    # df = handle_dates_features(df)
    # df = handle_categorical_cols(df)
    # df = handle_ivi(df)
    df = handle_ki67(df)


if __name__ == '__main__':
    main()
