import pandas as pd
import numpy as np

# ML Pkgs
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Split Dataset into Train and Text
from sklearn.model_selection import train_test_split
# Feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Multi Label Pkgs
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
import skmultilearn


# Convert Our Multi-Label Prob to Multi-Class


def build_model(model, mlb_estimator):
    # Create an Instance
    from sklearn.preprocessing import MinMaxScaler  # fixed import

    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    clf = mlb_estimator(model)
    # clf.fit(X_train, y_train)

    return clf


def get_models():
    # binary_relevance
    clf_RF_binary_model = build_model(RandomForestClassifier(), BinaryRelevance)
    clf_KNN_binary_model = build_model(KNeighborsClassifier(), BinaryRelevance)
    clf_LR_binary_model = build_model(LogisticRegression(), BinaryRelevance)
    clf_DR_binary_model = build_model(DecisionTreeClassifier(), BinaryRelevance)

    # Chains:
    clf_RF_chain_model = build_model(RandomForestClassifier(), ClassifierChain)
    clf_KNN_chain_model = build_model(KNeighborsClassifier(), ClassifierChain)
    clf_LR_chain_model = build_model(LogisticRegression(), ClassifierChain)
    clf_DR_chain_model = build_model(DecisionTreeClassifier(), ClassifierChain)

    # PowerSet:
    clf_RF_PowerSet_model = build_model(RandomForestClassifier(), LabelPowerset)
    clf_KNN_PowerSet_model = build_model(KNeighborsClassifier(), LabelPowerset)
    clf_LR_PowerSet_model = build_model(LogisticRegression(), LabelPowerset)
    clf_DR_PowerSet_model = build_model(DecisionTreeClassifier(), LabelPowerset)

    # return clf_chain_model, clf_labelPS_model
    # --------------------------------------adaptive algorithms: KNN RandomForest --------------------------------------

    # -------------------------------------------assemble methods -----------------------------------------------
    return clf_RF_chain_model, clf_KNN_chain_model, clf_LR_chain_model, clf_DR_chain_model, \
           clf_RF_PowerSet_model, clf_KNN_PowerSet_model, clf_LR_PowerSet_model, clf_DR_PowerSet_model,\
           clf_RF_binary_model , clf_KNN_binary_model, clf_LR_binary_model, clf_DR_binary_model
