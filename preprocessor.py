import pandas as pd
import numpy as np
import utilities


def handle_ordered_categorical_cols(df):
    # 'Histological Diagnosis'
    pass


def handle_categorical_cols(df, encoder=True):
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    # 'Form Name'
    categorical_cols = [' Form Name', ' Hospital', 'אבחנה-Histological diagnosis',
                        'אבחנה-Margin Type']  # TODO 'אבחנה-Basic stage',

    if encoder:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        encoder.fit(df[categorical_cols])
    transformed = encoder.transform(df[categorical_cols])
    transformed_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out(categorical_cols))
    result = pd.concat([df, transformed_df], axis=1)
    return result.drop(categorical_cols, axis=1)


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
    utilities.present_unique_values(df, col_name='אבחנה-Ivi -Lymphovascular invasion')
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
    utilities.present_unique_values(df, 'אבחנה-KI67 protein')
    low = ['6-9%', '15%', '20', '20%', '6%-9%', 'score 3-4', '15-20%', '10%', '5% vs 16-30%', '10', '9', '5%', '8-15%',
           '1', '<5%', '15', '4%', '1%', '5-10%', '10-14%', '10-15%', '15-30%', '3%', '2%', '1-2%', '10-20%', '06-Sep',
           '03-May', '5']
    medium = [
        '50', '45%', '455', '40%', '30%', '40=50%', '50%', '20-30%', '30', '10-49%', '30-50%', '45', '40', '50% score4',
        'score 3', '49%', '40-50%', '25%', '30-40%', '35%', '30-35%']
    medium_high = ['60', '60%', '50-60%', '70', '50-70%', '+>50%', '60-70%']
    high = ['90%', '80%', '70%', '90', '95%', '75%', '80-90%', '85%', 'High', 'Score 4']

    for val in low:
        cur = df['אבחנה-KI67 protein'] == val
        indices = cur[cur].index
        df.loc[indices, 'אבחנה-KI67 protein'] = 20

    for val in medium:
        cur = df['אבחנה-KI67 protein'] == val
        indices = cur[cur].index
        df.loc[indices, 'אבחנה-KI67 protein'] = 40

    for val in medium_high:
        cur = df['אבחנה-KI67 protein'] == val
        indices = cur[cur].index
        df.loc[indices, 'אבחנה-KI67 protein'] = 60

    for val in high:
        cur = df['אבחנה-KI67 protein'] == val
        indices = cur[cur].index
        df.loc[indices, 'אבחנה-KI67 protein'] = 80

    # todo: decide based on correlation nan values
    nan_idx = df[df['אבחנה-KI67 protein'].isnull()].index.tolist()
    df['אבחנה-KI67 protein'][nan_idx] = 10
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
