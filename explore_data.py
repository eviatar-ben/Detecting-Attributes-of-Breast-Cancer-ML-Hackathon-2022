import math
import pandas as pd
import re
import numpy as np


def preprocessing(features: pd.DataFrame):
    features["Her2_processed"] = features["אבחנה-Her2"].apply(processing_her2)

    features["er_processed"] = features["אבחנה-er"].apply(processing_err)

    features["pr_processed"] = features["אבחנה-pr"].apply(processing_err)




r_num = "\d+\.*\d*"
r_neg = "[Hnm,][erf][gfc]|[as][kj][ghj][ka]h|^[-=_]|^n[od]|\(-\)|שלילי"
r_strong = "strong|חזק"
r_weak = "weak|חלש"
r_percent = "%"
r_pos = "jhuch|חיובי|po|\+|strong|weak|חזק|חלש"
r_mid = "equi|\?|inde|inter|בינוני"
r_zero = "[0_o\)]"


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

def processing_her2(string):  # todo: change!
    if type(string) == str:
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
                    return 0

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
    preprocessing(pd.read_csv("splited_datasets/features_train_base_0.csv"))
