import sys
sys.path.append('../Utils')

from Utils.clas_plot import *
import os

import random
import scipy
from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scikitplot as skplt
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn import datasets 
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import *
from sklearn.metrics import *
# Import necessary libraries for new classifiers
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# ... (previous code) ...

# Add a new function for Dummy Classifier
def train_DummyClassifier(X_train, y_train):
    print('Training DummyClassifier ...')
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    return dummy

# Modify the train_DecisionTree function to use specified hyperparameters for classification
def train_DecisionTreeClassifier(X_train, y_train):
    print('Training DecisionTreeClassifier ...')
    tree = DecisionTreeClassifier(random_state=0)
    param_distributions = {
        'criterion': ["gini", "entropy"],
        'max_depth': [2, 3, 4]
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        tree, param_distributions, n_iter=30, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

# Add a new function for Logistic Regression
def train_LogisticRegressionClassifier(X_train, y_train):
    print('Training LogisticRegressionClassifier ...')
    lr = LogisticRegression(solver='liblinear', multi_class='auto', random_state=0)
    lr.fit(X_train, y_train)
    return lr

# Modify the train_RandomForest function to use specified hyperparameters for classification
def train_RandomForestClassifier(X_train, y_train):
    print('Training RandomForestClassifier ...')
    forest = RandomForestClassifier(random_state=0)
    param_distributions = {
        'criterion': ["gini", "entropy"],
        'max_features': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_samples_split': [10, 20, 30, 40, 50],
        'max_depth': [4, 6, 8, 10, 12],
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        forest, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

# Modify the train_RandomForest function to use specified hyperparameters for classification
def train_ExtraTreesClassifier(X_train, y_train):
    print('Training ExtraTreesClassifier ...')
    extra_trees = ExtraTreesClassifier(random_state=0)
    param_distributions = {
        'criterion': ["gini", "entropy"],
        'max_features': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_samples_split': [10, 20, 30, 40, 50],
        'max_depth': [4, 6, 8, 10, 12],
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        extra_trees, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

# Add a new function for XGBoost Multi-Class Classification
def train_XGBoostMultiClassClassifier(X_train, y_train):
    print('Training XGBoost Multi-Class Classifier ...')
    xgb = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=0)
    param_distributions = {
        'eta': [0.05, 0.075, 0.1, 0.15],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        xgb, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

# Add a new function for CatBoost Classifier
def train_CatBoostClassifier(X_train, y_train):
    print('Training CatBoost Classifier ...')
    catboost = CatBoostClassifier(iterations=50, verbose=0, random_state=0)
    param_distributions = {
        'learning_rate': [0.05, 0.1, 0.2],
        'depth': [2, 3, 4, 5, 6],
        'rsm': [0.7, 0.8, 0.9, 1],
        'subsample': [0.7, 0.8, 0.9, 1],
        'min_data_in_leaf': [1, 5, 10, 15, 20, 30, 50],
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        catboost, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

# Modify the train_KNN function to use specified hyperparameters for classification
def train_KNeighborsClassifier(X_train, y_train):
    print('Training KNeighborsClassifier ...')
    knn = KNeighborsClassifier()
    param_distributions = {
        'n_neighbors': [3, 5, 7],
        'weights': ["uniform", "distance"]
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        knn, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

# classifiers
# def train_KNN(X_train, y_train):
#     print('Training KNN ...')
#     knn = KNeighborsClassifier()
# #     scoring = ['roc_auc']
#     param_distributions = {
#         'n_neighbors' : scipy.stats.randint(1,20)
#     }
#     randcv = sklearn.model_selection.RandomizedSearchCV(
#         knn,param_distributions,n_iter=20,cv=3,n_jobs=4,random_state=0)
#     randcv.fit(X_train, y_train)
#     return randcv.best_estimator_

def train_SVM(X_train, y_train):
    print('Training SVM ...')
    svm = SVC(kernel='rbf', probability=True, cache_size=3000, random_state=0)
#     scoring = ['roc_auc']
    param_distributions = {
        'C' : scipy.stats.reciprocal(1.0, 100.),
        'gamma' : scipy.stats.reciprocal(0.01, 10.),
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        svm,param_distributions,n_iter=20,cv=3,n_jobs=4,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

# def train_DecisionTree(X_train, y_train):
#     print('Training DecisionTree ...')
#     tree = DecisionTreeClassifier(random_state=0)
# #     scoring = ['roc_auc']
#     param_distributions = {
#         'max_depth' : scipy.stats.randint(10,1000)
#     }
#     randcv = sklearn.model_selection.RandomizedSearchCV(
#         tree,param_distributions,n_iter=30,cv=3,n_jobs=-1,random_state=0)
#     randcv.fit(X_train, y_train)
#     return randcv.best_estimator_

# def train_RandomForest(X_train, y_train):
#     print('Training RandomForest ...')
#     forest = RandomForestClassifier(random_state=0)
# #     scoring = ['roc_auc']
#     param_distributions = {
#         'max_depth' : scipy.stats.randint(10,100),
#         'n_estimators' : scipy.stats.randint(100,1000)
#     }
#     randcv = sklearn.model_selection.RandomizedSearchCV(
#         forest,param_distributions,n_iter=10,cv=3,n_jobs=-1,random_state=0)
#     randcv.fit(X_train, y_train)
#     return randcv.best_estimator_

def train_AdaBoost(X_train, y_train):
    print('Training AdaBoost ...')
    boost = AdaBoostClassifier(random_state=0)
#     scoring = ['roc_auc']
    param_distributions = {
        'learning_rate' : scipy.stats.uniform(0.75, 1.25),
        'n_estimators' : scipy.stats.randint(40,70)
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        boost,param_distributions,n_iter=30,cv=3,n_jobs=-1,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_    

def train_LogisticRegression(X_train, y_train):
    print('Training LogisticRegression ...')
    lr = LogisticRegression(solver='liblinear', multi_class='auto', random_state=0)
#     scoring = ['roc_auc']
    param_distributions = {
        'C' : scipy.stats.reciprocal(1.0, 1000.),
        'max_iter' : scipy.stats.randint(100,1000)
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        lr,param_distributions,n_iter=30,cv=3,n_jobs=-1,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_ 

def train_GaussianNaiveBayes(X_train, y_train):
    print('Training GaussianNaiveBayes ...')
    gaussian = GaussianNB()
#     scoring = ['roc_auc']
    param_distributions = {
        'var_smoothing' : scipy.stats.uniform(1e-10, 1e-9),
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        gaussian,param_distributions,n_iter=30,cv=3,n_jobs=-1,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_ 

def train_NeuralNetwork(X_train, y_train):
    print('Training NeuralNetwork ...')
    nn = MLPClassifier(solver='adam', random_state=0)
#     scoring = ['roc_auc']
    param_distributions = {
        'hidden_layer_sizes' : [(100,50,10)],
        'learning_rate_init' : scipy.stats.uniform(0.001, 0.005),
        'max_iter' : scipy.stats.randint(200,500)
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        nn,param_distributions,n_iter=10,cv=3,n_jobs=-1,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_ 


def run_all_clfs(X_train, y_train, X_test, y_test):
    all_clfs = []
    clf_names = []

    clf0 = train_DummyClassifier(X_train, y_train)
    all_clfs.append(clf1)
    clf_names.append('Dummy Classifier')

    # clf1 = train_KNN(X_train, y_train)
    # all_clfs.append(clf1)
    # clf_names.append('KNN')

    clf2 = train_SVM(X_train, y_train)
    all_clfs.append(clf2)
    clf_names.append('SVM')

    # clf3 = train_DecisionTree(X_train, y_train)
    # all_clfs.append(clf3)
    # clf_names.append('Decision Tree')

    # clf4 = train_RandomForest(X_train, y_train)
    # all_clfs.append(clf4)
    # clf_names.append('Random Forest')

    clf5 = train_AdaBoost(X_train, y_train)
    all_clfs.append(clf5)
    clf_names.append('AdaBoost')

    clf6 = train_LogisticRegression(X_train, y_train)
    all_clfs.append(clf6)
    clf_names.append('Logistic regression')

    clf7 = train_GaussianNaiveBayes(X_train, y_train)
    all_clfs.append(clf7)
    clf_names.append('Gaussian Naive Bayes')

    clf8 = train_NeuralNetwork(X_train, y_train)
    all_clfs.append(clf8)
    clf_names.append('NeuralNetwork')

    clf9 = train_DecisionTreeClassifier(X_train, y_train)
    all_clfs.append(clf9)
    clf_names.append('Decision Tree Classifier')

    clf10 = train_LogisticRegressionClassifier(X_train, y_train)
    all_clfs.append(clf10)
    clf_names.append('Logistic Regression Classifier')

    clf11 = train_RandomForestClassifier(X_train, y_train)
    all_clfs.append(clf11)
    clf_names.append('Random Forest Classifier')

    clf12 = train_ExtraTreesClassifier(X_train, y_train)
    all_clfs.append(clf12)
    clf_names.append('Extra Trees Classifier')

    clf14 = train_XGBoostMultiClassClassifier(X_train, y_train)
    all_clfs.append(clf14)
    clf_names.append('XGBoost Multi-Class Classifier')

    clf15 = train_CatBoostClassifier(X_train, y_train)
    all_clfs.append(clf15)
    clf_names.append('CatBoost Classifier')

    clf16 = train_KNeighborsClassifier(X_train, y_train)
    all_clfs.append(clf16)
    clf_names.append('KNeighbors Classifier')

    return all_clfs, clf_names

