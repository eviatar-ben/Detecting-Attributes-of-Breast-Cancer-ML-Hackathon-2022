import pandas as pd


def handle_ordered_categorical_cols(df):
    # 'Histological Diagnosis'
    pass


def handle_categorical_cols(df):
    # 'Form Name'
    df[' Form Name'] = df[' Form Name'].astype('category')
    # ' Hospital'
    # todo: consider to drop this col
    df[' Hospital'] = df[' Hospital'].astype('category')
    # 'Basic stage

    # 'Histological Diagnosis


def drop_cols(df):
    # User name
    print(len(df))
    df.drop('User Name', axis=1, inplace=True)
    print(len(df))


def main():
    df = pd.read_csv(r'splited_datasets/features_train_base_0.csv')
    handle_categorical_cols(df)
    drop_cols(df)


if __name__ == '__main__':
    main()
