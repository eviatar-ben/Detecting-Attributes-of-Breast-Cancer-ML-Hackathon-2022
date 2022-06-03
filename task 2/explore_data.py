import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from pandas import CategoricalDtype  # TODO: pd.CategoricalDtype instead
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from preprocessor import *


def preprocessing(data: pd.DataFrame):
    scaler = StandardScaler(copy=False, with_mean=False)
    features = data

    features["Her2_processed"] = features["אבחנה-Her2"].apply(processing_her2)

    features["er_processed"] = features["אבחנה-er"].apply(processing_err)

    features["pr_processed"] = features["אבחנה-pr"].apply(processing_err)

    features["lymph nodes mark processed"] = \
        features["אבחנה-N -lymph nodes mark (TNM)"].fillna("NX").apply(processing_TNM, args="n")

    features["metastases mark processed"] = \
        features["אבחנה-M -metastases mark (TNM)"].fillna("MX").apply(processing_TNM, args="m")

    features["Tumor mark processed"] = \
        features["אבחנה-T -Tumor mark (TNM)"].fillna("TX").apply(processing_TNM, args="t")

    features["stage processed"] = features["אבחנה-Stage"].fillna("ex").apply(processing_TNM, args="e")

    features["time from first surgery processed"] = np.zeros(features["אבחנה-Surgery date1"].size)

    features["time from second surgery processed"] = np.zeros(features["אבחנה-Surgery date1"].size)

    features["time from third surgery processed"] = np.zeros(features["אבחנה-Surgery date1"].size)

    features["time from first surgery processed"] = scaler.fit_transform(
        features.apply(process_dates, axis=1).to_numpy().reshape(-1, 1))

    features["time from second surgery processed"] = scaler.fit_transform(
        features.apply(process_dates_2, axis=1).to_numpy().reshape(-1, 1))

    features["time from third surgery processed"] = scaler.fit_transform(
        features.apply(process_dates_3, axis=1).to_numpy().reshape(-1, 1))

    data.update(features)


r_date = "\d+/\d+/\d+"


def process_dates(data):
    if isinstance(data["אבחנה-Surgery date1"], str) and not (data["אבחנה-Diagnosis date"] is np.nan):
        if re.findall(r_date, data["אבחנה-Surgery date1"]):
            date = datetime.strptime(data["אבחנה-Surgery date1"], "%d/%m/%Y")
            # print((data["אבחנה-Diagnosis date"] - date).days)
            return np.maximum((data["אבחנה-Diagnosis date"] - date).days, 0)

    return 0


def process_dates_2(data):
    if isinstance(data["אבחנה-Surgery date2"], str) and not (data["אבחנה-Diagnosis date"] is np.nan):
        if re.findall(r_date, data["אבחנה-Surgery date2"]):
            date = datetime.strptime(data["אבחנה-Surgery date2"], "%d/%m/%Y")
            return np.maximum((data["אבחנה-Diagnosis date"] - date).days, 0)

    return 0


def process_dates_3(data):
    if isinstance(data["אבחנה-Surgery date3"], str) and not (data["אבחנה-Diagnosis date"] is np.nan):
        if re.findall(r_date, data["אבחנה-Surgery date3"]):
            date = datetime.strptime(data["אבחנה-Surgery date3"], "%d/%m/%Y")
            return np.maximum((data["אבחנה-Diagnosis date"] - date).days, 0)

    return 0


r_num = "\d+\.*\d*"
r_neg = "[Hnm,][erf][gfc]|[as][kj][ghj][ka]h|^[-=_]|^n[od]|non|\(-\)|שלילי|low"
r_strong = "strong|חזק"
r_weak = "weak|חלש"
r_percent = "%"
r_pos = "jhuch|חיובי|po|\+|strong|weak|חזק|חלש"
r_mid = "equi|\?|inde|inter|בינוני|border"
r_zero = "[0_o\)]"


def processing_TNM(string, char):
    r_tnm = fr"{char}\d"
    match = re.findall(r_tnm, string, re.IGNORECASE)

    if match:
        return int(match[0][1])
    elif string == "MF" or string == "Tis":
        return 1
    return -10


def processing_err(string):
    if type(string) == str:
        if re.findall(r_pos, string, re.IGNORECASE):
            if re.findall(r_strong, string, re.IGNORECASE):
                return 4
            elif re.findall(r_weak, string, re.IGNORECASE):
                return 2
            if re.findall(r_num, string, re.IGNORECASE):
                return process_nums(string)
            else:
                return 3
        elif re.findall(r_neg, string, re.IGNORECASE):
            return 1

        elif re.findall(r_mid, string, re.IGNORECASE):
            return 2

        if re.findall(r_num, string, re.IGNORECASE):
            return process_nums(string)

        return -10
    else:
        return -10


def process_nums(string):
    percent = (re.findall(r_percent, string) is True)
    for num in re.findall(r_num, string, re.IGNORECASE):
        num = float(num)

        if percent:
            if num < 10:
                return 1
            elif num < 33:
                return 2
            elif num < 66:
                return 3
            else:
                return 4

        else:
            if num < 3:
                return 1
            elif num < 4:
                return 2
            elif num < 6:
                return 3
            else:
                return 4


def processing_her2(string):
    if type(string) == str:
        if string.lower() == "amplified":
            return 3
        nums = []
        for num in re.findall(r_num, string, re.IGNORECASE):
            num = float(num)
            if num <= 10:
                if num > 2.2:
                    return 3
                elif num > 1.8:
                    return 2
                elif num > 0:
                    return 1
                else:
                    return -10

            nums.append(num)

        if re.findall(r_pos, string, re.IGNORECASE):
            return 3
        elif re.findall(r_neg, string, re.IGNORECASE):
            return 1

        elif re.findall(r_mid, string, re.IGNORECASE):
            return 2

        if nums:
            if np.min(nums) > 10:
                return 3
            return np.min(nums)

        if re.findall(r_zero, string, re.IGNORECASE) and (len(string) == 1):
            return 0

        return -10
    else:
        return -10


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
                   'אבחנה-Ivi -Lymphovascular invasion',
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
                   'אבחנה-Surgery date1',
                   'אבחנה-Surgery date2',
                   'אבחנה-Surgery date3',
                   'surgery before or after-Activity date',
                   'אבחנה-Diagnosis date',
                   "אבחנה-Basic stage",
                   "אבחנה-Histopatological degree",
                   "אבחנה-Lymphatic penetration",
                   ' Hospital',
                   'אבחנה-Margin Type',
                   ' Form Name',
                   'User Name'
                   ])

    return df, num_imp, ord_imp, encoder


def handle_ordered_categories(df: pd.DataFrame, imputers=None) -> Iterable[SimpleImputer]:
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


def handle_numerical(df: pd.DataFrame, imputers=None) -> Iterable[SimpleImputer]:
    df['אבחנה-Surgery sum'].mask(
        (df['אבחנה-Surgery sum'].isna()),
        (df['אבחנה-Surgery date1'].notna()) +
        (df['אבחנה-Surgery date2'].notna()) +
        (df['אבחנה-Surgery date3'].notna()),
        inplace=True
    )
    df['אבחנה-Surgery sum'] = df['אבחנה-Surgery sum'].astype(float)

    df['אבחנה-Nodes exam'].fillna(0, inplace=True)
    df['אבחנה-Positive nodes'].mask(
        (df["אבחנה-Positive nodes"].isna()), df['אבחנה-Nodes exam'],
        inplace=True
    )

    median_imputer = SimpleImputer(strategy="median")
    if imputers is not None:
        median_imputer = imputers[0]

    df["אבחנה-Age"] = median_imputer.fit_transform(df[["אבחנה-Age"]])

    return [median_imputer]

def load_data_part_1(train_X_fn: Path, train_y_fn: Path):
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
    # full_data = full_data.loc[:, ~full_data.columns.str.contains('^Unnamed')]
    full_data.reset_index(inplace=True, drop=True)
    return full_data

if __name__ == '__main__':
    preprocessing(pd.read_csv("train.feats.csv", parse_dates=[
        "אבחנה-Diagnosis date",
        "אבחנה-Surgery date1",
        "אבחנה-Surgery date2",
        "אבחנה-Surgery date3"
    ], infer_datetime_format=True, dayfirst=True))
