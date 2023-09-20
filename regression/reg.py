import sys
sys.path.append('../Utils')
from Utils.pre_processing import *
from Utils.rg_plot import *

import random
import scipy
from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scikitplot as skplt
import sklearn
from sklearn import preprocessing
from sklearn import datasets 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct,WhiteKernel,RBF,Matern,RationalQuadratic,ExpSineSquared,ConstantKernel,PairwiseKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA
from sklearn.model_selection import *
from sklearn.metrics import *

# scipy.stats.randint(1,20)
# scipy.stats.reciprocal(1.0, 100.),
# scipy.stats.uniform(0.75, 1.25),

DATA_PATH = 'dataset/'
IMAGE_PATH = 'img/'


from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

# ... (previous code) ...

def train_DummyRegressor(X_train, y_train):
    print('Training DummyRegressor ...')
    dummy_regressor = DummyRegressor(strategy="mean")
    dummy_regressor.fit(X_train, y_train)
    return dummy_regressor

def train_DecisionTree(X_train, y_train):
    print('Training DecisionTree ...')
    dt = DecisionTreeRegressor()
    param_distributions = {
        "criterion": ['absolute_error', 'squared_error', 'friedman_mse', 'poisson'],
        "max_depth": [2, 3, 4]
    }
    randcv = RandomizedSearchCV(dt, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv

def train_LinearRegression(X_train, y_train):
    print('Training LinearRegression ...')
    linear = LinearRegression(n_jobs=-1)
    # No hyperparameters to tune for Linear Regression
    linear.fit(X_train, y_train)
    return linear

def train_RandomForest(X_train, y_train):
    print('Training RandomForest ...')
    rf = RandomForestRegressor()
    param_distributions = {
        "criterion": ['absolute_error', 'squared_error', 'friedman_mse', 'poisson'],
        "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [10, 20, 30, 40, 50],
        "max_depth": [4, 6, 8, 10, 12],
    }
    randcv = RandomizedSearchCV(rf, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv

def train_ExtraTrees(X_train, y_train):
    print('Training ExtraTrees ...')
    et = ExtraTreesRegressor()
    param_distributions = {
        "criterion": ['absolute_error', 'squared_error', 'friedman_mse', 'poisson'],
        "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [10, 20, 30, 40, 50],
        "max_depth": [4, 6, 8, 10, 12],
    }
    randcv = RandomizedSearchCV(et, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv

def train_XGBRegressor(X_train, y_train):
    print('Training XGBoost ...')
    xgb = XGBRegressor()
    param_distributions = {
        "objective": ["reg:squarederror"],
        "eval_metric": ["rmse"],
        "eta": [0.05, 0.075, 0.1, 0.15],
        "max_depth": [1, 2, 3, 4],
        "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    randcv = RandomizedSearchCV(xgb, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv

def train_CatBoostRegressor(X_train, y_train):
    print('Training CatBoost ...')
    catboost = CatBoostRegressor(silent=True)
    param_distributions = {
        "learning_rate": [0.05, 0.1, 0.2],
        "depth": [2, 3, 4, 5, 6],
        "rsm": [0.7, 0.8, 0.9, 1],  # random subspace method # random subspace method
        "min_data_in_leaf": [1, 5, 10, 15, 20, 30, 50],
    }
    randcv = RandomizedSearchCV(catboost, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv

def train_KNeighborsRegressor(X_train, y_train):
    print('Training KNeighborsRegressor ...')
    knn = KNeighborsRegressor()
    param_distributions = {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    }
    randcv = RandomizedSearchCV(knn, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=0)
    randcv.fit(X_train, y_train)
    return randcv

def train_SVR(X_train, y_train):
    print('Training SVR ...')
    svr = SVR()
    param_distributions = {
        'kernel' : [DotProduct(),WhiteKernel(),RBF(),Matern(),RationalQuadratic()],
        'C' : scipy.stats.reciprocal(1.0, 10.),
#         'epsilon' : scipy.stats.uniform(0.1, 0.5),
#         'gamma' : scipy.stats.reciprocal(0.01, 0.1),
    }
    randcv = RandomizedSearchCV(svr,param_distributions,n_iter=10,cv=3,n_jobs=1,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv

# def train_DecisionTree(X_train, y_train):
#     print('Training DecisionTree ...')
#     tree = DecisionTreeRegressor(random_state=0)
#     param_distributions = {
#         'max_depth' : scipy.stats.randint(10,100)
#     }
#     randcv = sklearn.model_selection.RandomizedSearchCV(tree,param_distributions,n_iter=30,cv=3,n_jobs=1,random_state=0)
#     randcv.fit(X_train, y_train)
#     return randcv

# def train_RandomForest(X_train, y_train):
#     print('Training RandomForest ...')
#     forest = RandomForestRegressor(random_state=0, warm_start=True)
#     param_distributions = {
#         'max_depth' : scipy.stats.randint(1,50),
#         'n_estimators' : scipy.stats.randint(100,200)
#     }
#     randcv = sklearn.model_selection.RandomizedSearchCV(forest,param_distributions,n_iter=10,cv=3,random_state=0)
#     randcv.fit(X_train, y_train)
#     return randcv

def train_AdaBoost(X_train, y_train):
    print('Training AdaBoost ...')
    boost = AdaBoostRegressor(random_state=0)
    param_distributions = {
        'loss' : ['linear', 'square', 'exponential'],
        'learning_rate' : scipy.stats.uniform(0.75, 1.25),
        'n_estimators' : scipy.stats.randint(40,100)
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(boost,param_distributions,n_iter=10,cv=3,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv

def train_GaussianProcess(X_train, y_train):
    print('Training GaussianProcess ...')
    alpha = 1e-9
    while(True):
        try:
            gaussian = GaussianProcessRegressor(normalize_y=True, random_state=0, optimizer=None, alpha=alpha)
            param_distributions = {
                'kernel' : [DotProduct(),WhiteKernel(),RBF(),Matern(),RationalQuadratic()],
                'n_restarts_optimizer' : scipy.stats.randint(0,10),
        #         'alpha' : scipy.stats.uniform(1e-9, 1e-8)
            }
            randcv = sklearn.model_selection.RandomizedSearchCV(gaussian,param_distributions,n_iter=5,cv=3,random_state=0)
            randcv.fit(X_train, y_train)
            return randcv
        except:
            alpha *= 10

def train_LinearRegression(X_train,y_train):
    print('Training LinearRegression ...')
    linear = LinearRegression(n_jobs=-1)
    param_distributions = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'positive': [True, False]}

    randcv = sklearn.model_selection.RandomizedSearchCV(linear,param_distributions,n_iter=2,cv=3,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv

def train_NeuralNetwork(X_train, y_train):
    print('Training NeuralNetwork ...')
    nn = MLPRegressor(random_state=0, warm_start=True)
    param_distributions = {
        'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        'solver' : ['lbfgs', 'adam'],
        'hidden_layer_sizes' : [(100,50,25),(200,100,50)],
        'learning_rate_init' : scipy.stats.uniform(0.001, 0.005),
        'max_iter' : scipy.stats.randint(200,500)
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(nn,param_distributions,n_iter=5,cv=3,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv

def run_all_regrs(X_train, y_train, X_test, y_test):
    all_regrs = []
    regr_names = []

    regr0 = train_DummyRegressor(X_train, y_train)
    all_regrs.append(regr0)
    regr_names.append('Dummy Classifier')

    regr1 = train_SVR(X_train, y_train)
    all_regrs.append(regr1.best_estimator_)
    regr_names.append('SVR')

    # regr2 = train_DecisionTree(X_train, y_train)
    # all_regrs.append(regr2.best_estimator_)
    # regr_names.append('Decision Tree')

    # regr3 = train_RandomForest(X_train, y_train)
    # all_regrs.append(regr3.best_estimator_)
    # regr_names.append('Random Forest')

    regr4 = train_AdaBoost(X_train, y_train)
    all_regrs.append(regr4.best_estimator_)
    regr_names.append('AdaBoost')

    regr5 = train_GaussianProcess(X_train, y_train)
    all_regrs.append(regr5.best_estimator_)
    regr_names.append('Gaussian Process')

    regr6 = train_LinearRegression(X_train, y_train)
    all_regrs.append(regr6)
    regr_names.append('Linear Regression')

    regr7 = train_NeuralNetwork(X_train, y_train)
    all_regrs.append(regr7.best_estimator_)
    regr_names.append('NeuralNetwork')

    regr8 = train_DecisionTree(X_train, y_train)
    all_regrs.append(regr8.best_estimator_)
    regr_names.append('Decision Tree Classifier')

    regr9 = train_RandomForest(X_train, y_train)
    all_regrs.append(regr9.best_estimator_)
    regr_names.append('Random Forest Classifier')

    regr10 = train_ExtraTrees(X_train, y_train)
    all_regrs.append(regr10.best_estimator_)
    regr_names.append('Extra Trees Classifier') 

    regr11 = train_XGBRegressor(X_train, y_train)
    all_regrs.append(regr11.best_estimator_)
    regr_names.append('XGBoost Multi-Class Classifier')

    regr12 = train_CatBoostRegressor(X_train, y_train)
    all_regrs.append(regr12.best_estimator_)
    regr_names.append('CatBoost Classifier')

    regr13 = train_KNeighborsRegressor(X_train, y_train)
    all_regrs.append(regr13.best_estimator_)
    regr_names.append('KNeighbors Classifier')

    regr14 = train_LinearRegression(X_train, y_train)
    all_regrs.append(regr14)
    regr_names.append('Logistic Regression Classifier')

    return all_regrs, regr_names

