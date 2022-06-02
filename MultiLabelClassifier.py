# Load EDA Pkgs
import pandas as pd
import numpy as np

# Load Data Viz Pkgs
import matplotlib.pyplot as plt
import seaborn as sns

# ML Pkgs
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, classification_report

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

# binary classification
def binary_relevance(X_train, y_train):
    """ basic approaches to multi-label classification, it ignores relationships between labels
    correlation between labels are lost """
    binary_rel_clf = BinaryRelevance(MultinomialNB())
    binary_rel_clf.fit(X_train, y_train)


def build_model(model, mlb_estimator, X_train, y_train):
    # Create an Instance
    from sklearn.preprocessing import MinMaxScaler  # fixed import

    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    clf = mlb_estimator(model)
    # clf.fit(X_train, y_train)
    return clf


def get_models(X_train, y_train):
    # binary_relevance(X_train, y_train)
    # Chains:
    clf_chain_model = build_model(RandomForestClassifier(), ClassifierChain, X_train, y_train)

    # PowerSet:
    clf_labelPS_model = build_model(RandomForestClassifier(), LabelPowerset, X_train, y_train)

    # return clf_chain_model, clf_labelPS_model
    # --------------------------------------adaptive algorithms: KNN RandomForest --------------------------------------

    # -------------------------------------------assemble methods -----------------------------------------------
    return clf_chain_model, clf_labelPS_model
