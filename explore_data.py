import math
import pandas as pd
import re
import numpy as np


def preprocessing():
    features = pd.read_csv("splited_datasets/features_train_base_0.csv")
    features["Her2_processed"] = features["אבחנה-Her2"].apply(processing_her2).fillna(0)
    print(features["אבחנה-er"].unique())

    features["er_processed"] = features["אבחנה-er"].apply(processing_err)


r_num = "\d+\.*\d*"
r_neg = "[Hnm,][erf][gfc]|[as][kj][ghj][ka]h|^[-=_]|^n[od]|\(-\)|שלילי"
r_pos = "jhuch|חיובי|po|\+"
r_mid = "equi|\?|inde|inter|בינוני"
r_zero = "[0_o\)]"


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

        # nums = []
        # for num in re.findall(r_num, string, re.IGNORECASE):
        #     num = float(num)
        #     if num <= 3 and num.is_integer():
        #         return num
        #
        #     nums.append(num)
        if nums:
            if np.min(nums) > 10:
                return 3
            return np.min(nums)

        if re.findall(r_zero, string, re.IGNORECASE) and (len(string) == 1):
            return 0

        return np.nan
    else:
        return string


def processing_err(string):
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

        # nums = []
        # for num in re.findall(r_num, string, re.IGNORECASE):
        #     num = float(num)
        #     if num <= 3 and num.is_integer():
        #         return num
        #
        #     nums.append(num)
        if nums:
            if np.min(nums) > 10:
                return 3
            return np.min(nums)

        if re.findall(r_zero, string, re.IGNORECASE) and (len(string) == 1):
            return 0

        return np.nan
    else:
        return string


if __name__ == '__main__':
    preprocessing()
