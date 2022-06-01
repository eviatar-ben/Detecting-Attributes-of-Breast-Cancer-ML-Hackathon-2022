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

# load data
X_train = pd.read_csv(r'splited_datasets/features_train_base_0.csv')
y_train = pd.read_csv(r'splited_datasets/labels_train_base_0.csv')
# Convert Our Multi-Label Prob to Multi-Class
# binary classficiation
binary_rel_clf = BinaryRelevance(MultinomialNB())
binary_rel_clf.fit(X_train, y_train)
