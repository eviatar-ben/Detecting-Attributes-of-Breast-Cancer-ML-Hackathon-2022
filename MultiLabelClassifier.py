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

# load data
X_train = pd.read_csv(r'splited_datasets/features_train_base_0.csv')
y_train = pd.read_csv(r'splited_datasets/labels_train_base_0.csv')


# Convert Our Multi-Label Prob to Multi-Class

# binary classification
def binary_relevance():
    """ basic approaches to multi-label classification, it ignores relationships between labels
    correlation between labels are lost """
    binary_rel_clf = BinaryRelevance(MultinomialNB())
    binary_rel_clf.fit(X_train, y_train)
    br_prediction = binary_rel_clf.predict(X_test)
    # Accuracy
    accuracy_score(y_test, br_prediction)


def build_model(model, mlb_estimator, xtrain, ytrain, xtest, ytest):
    # Create an Instance
    clf = mlb_estimator(model)
    clf.fit(xtrain, ytrain)
    # Predict
    clf_predictions = clf.predict(xtest)
    # Check For Accuracy
    acc = accuracy_score(ytest, clf_predictions)
    ham = hamming_loss(ytest, clf_predictions)
    result = {"accuracy:": acc, "hamming_score": ham}
    return result


# Chains:
clf_chain_model = build_model(MultinomialNB(), ClassifierChain, X_train, y_train, X_test, y_test)

# Powerset:
clf_labelP_model = build_model(MultinomialNB(), LabelPowerset, X_train, y_train, X_test, y_test)

# -------------------------------------------adaptive algorithms: KNN RF -----------------------------------------------


# -------------------------------------------ansamble methods -----------------------------------------------
