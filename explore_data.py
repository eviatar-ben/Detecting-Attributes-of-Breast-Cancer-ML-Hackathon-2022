import pandas as pd
import re
import numpy as np
from datetime import datetime


def preprocessing(features: pd.DataFrame):
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

    features["time from first surgery processed"] = features.apply(process_dates, axis=1)

    features["time from second surgery processed"] = features.apply(process_dates_2, axis=1)

    features["time from third surgery processed"] = features.apply(process_dates_3, axis=1)


r_date = "\d+/\d+/\d+"


def process_dates(data):
    if isinstance(data["אבחנה-Surgery date1"], str) and not (data["אבחנה-Diagnosis date"] is np.nan):
        if re.findall(r_date, data["אבחנה-Surgery date1"]):
            date = datetime.strptime(data["אבחנה-Surgery date1"], "%d/%m/%Y")
            return (data["אבחנה-Diagnosis date"] - date).days

        else:
            return 0
    return 0


def process_dates_2(data):
    if isinstance(data["אבחנה-Surgery date2"], str) and not (data["אבחנה-Diagnosis date"] is np.nan):
        if re.findall(r_date, data["אבחנה-Surgery date2"]):
            date = datetime.strptime(data["אבחנה-Surgery date2"], "%d/%m/%Y")
            return (data["אבחנה-Diagnosis date"] - date).days

        else:
            return 0
    return 0


def process_dates_3(data):
    if isinstance(data["אבחנה-Surgery date3"], str) and not (data["אבחנה-Diagnosis date"] is np.nan):
        if re.findall(r_date, data["אבחנה-Surgery date3"]):
            date = datetime.strptime(data["אבחנה-Surgery date3"], "%d/%m/%Y")
            return (data["אבחנה-Diagnosis date"] - date).days

        else:
            return 0
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


if __name__ == '__main__':
    preprocessing(pd.read_csv("train.feats.csv", parse_dates=[
        "אבחנה-Diagnosis date",
        "אבחנה-Surgery date1",
        "אבחנה-Surgery date2",
        "אבחנה-Surgery date3"
    ], infer_datetime_format=True, dayfirst=True))
