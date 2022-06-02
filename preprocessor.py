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
                        'אבחנה-Margin Type']   # TODO 'אבחנה-Basic stage',

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


def main():
    df = pd.read_csv(r'splited_datasets/features_train_base_0.csv', parse_dates=[
        "אבחנה-Diagnosis date",
        "אבחנה-Surgery date1",
        "אבחנה-Surgery date2",
        "אבחנה-Surgery date3",
        "surgery before or after-Activity date"
    ], infer_datetime_format=True, dayfirst=True)
    drop_cols(df, ['User Name'])
    df = handle_dates_features(df)
    df = handle_categorical_cols(df)
    df = handle_ivi(df)





if __name__ == '__main__':
    main()
