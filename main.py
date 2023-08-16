import sys
sys.path.append('../Utils')
from Utils.pre_processing import *
from regression.reg import *
from classification.clas import *
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

DATA_PATH = 'dataset/'
IMAGE_PATH = 'img/'
def load_split_train_data(train_path, target_name, test_size=0.2, random_state=None):
    try:
        # Load train data
        if train_path.endswith('.csv'):
            train_data = pd.read_csv(train_path, sep=None, engine='python')
        elif train_path.endswith('.xls') or train_path.endswith('.xlsx'):
            train_data = pd.read_excel(train_path)
        elif train_path.endswith('.txt'):
            train_data = pd.read_csv(train_path, sep=None, engine='python')
        else:
            raise ValueError("Invalid file format for train data")
        
        # Check for header presence
        if not all(col in train_data.columns for col in train_data.columns):
            raise ValueError("Header missing in the dataset")
        
        print('Dataset: %s' % train_data.head())
        print('Shape: %s' % str(train_data.shape))
        check_class_distribution(train_data)
        train_data = replace_question_marks(train_data)

    
        # Separate features (X) and target (y)
        X = train_data.drop(columns=[target_name])
        y = train_data[target_name]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return X_train, y_train, X_test, y_test
    
    except FileNotFoundError:
        raise FileNotFoundError("File not found at the specified location")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the data. Check file format and delimiter")
    except Exception as e:
        raise e


mse_scores = dict()
r2_scores = dict()
clfs_all_roc_scores = dict()
clfs_all_pr_scores = dict()

#data sets
DATA_PATH = 'dataset/'
IMAGE_PATH = 'image/'
pd.set_option('display.max_rows', 100)

def main():
    print("Welcome to the ML Command-Line App!")

    
    print("\nChoose a model type:")
    print("1. Regression")
    print("2. Classification")
    print("0. Exit")
    choice = input("Enter your choice: ")

    if choice == "0":
        print("Exiting the app. Goodbye!")
        return 0
    elif choice == "1":
        model_type = "regression"
        train_file = input("Enter the path to the train dataset file: ")
        #test_file = input("Enter the path to the test dataset file: ")
        target = input("Enter the target column name: ")
        file_name = os.path.split(train_file)[-1]
        print(file_name)

        X_train, y_train, X_test, y_test = load_split_train_data(train_file, target)
        
        X_train, X_test = encode_labels(X_train,X_test)
        X_train, X_test = impute_value(X_train, X_test,'mean')
        X_train, X_test = normalize_data(X_train, X_test)
        X_train, X_test = dimension_reduction(X_train, X_test, n_components=15)
        
        all_regrs, regr_names = run_all_regrs(X_train, y_train, X_test, y_test)
        mse, r2 = plot_all(X_train, y_train, X_test, y_test, all_regrs, regr_names, file_name)
        
        for k,v in mse.items():
            if k not in mse_scores:
                mse_scores[k] = list()
            mse_scores[k].append(v)
        for k,v in r2.items():
            if k not in r2_scores:
                r2_scores[k] = list()
            r2_scores[k].append(v)
        
        # for k,v in mse_scores.items():
        #     print('%s-mse:'%k,["%.3f"%i for i in v])

        # for k,v in r2_scores.items():
        #     print('%s-r2:'%k,["%.3f"%i for i in v])
        
        for k,v in r2_scores.items():
            print('%s avg r2: %.2f, avg mse: %.2f'%(k,np.mean(v),np.mean(mse_scores[k])))

        best_model = None
        best_avg_r2 = float('-inf')
        best_avg_mse = float('inf')
        
        for model, r2_score in r2_scores.items():
            avg_r2 = sum(r2_score) / len(r2_score)
            avg_mse = sum(mse_scores[model]) / len(mse_scores[model])
            
            if avg_r2 > best_avg_r2 or (avg_r2 == best_avg_r2 and avg_mse < best_avg_mse):
                best_model = model
                best_avg_r2 = avg_r2
                best_avg_mse = avg_mse

        print(f"\nBest Regression Model: {best_model}")
        print(f"\nBest Average R2 Score: {best_avg_r2}")
        print(f"\nBest Average MSE Score: {best_avg_mse}")
        
            

    elif choice == "2":
        model_type = "classification"
        
        train_file = input("Enter the path to the train dataset file: ")
        #test_file = input("Enter the path to the test dataset file: ")
        target = input("Enter the target column name: ")
        file_name = os.path.split(train_file)[-1]
        print(file_name)
    

        X_train, y_train, X_test, y_test = load_split_train_data(train_file, target)

        X_train, X_test = encode_labels(X_train,X_test)
        X_train, X_test = dimension_reduction(X_train,X_test)

        X_train, X_test = impute_value(X_train, X_test,strategy = 'mean')
        X_train, X_test = normalize_data(X_train, X_test)

        y_train, y_test = encode_labels(y_train, y_test, -1)

        all_clfs, clf_names = run_all_clfs(X_train, y_train, X_test, y_test)
        roc_scores, pr_scores = plot_all(X_test, y_test, all_clfs, clf_names, file_name)

        for k,v in roc_scores.items():
            if k not in clfs_all_roc_scores:
                clfs_all_roc_scores[k] = list()
            clfs_all_roc_scores[k].append(v)
        for k,v in pr_scores.items():
            if k not in clfs_all_pr_scores:
                clfs_all_pr_scores[k] = list()
            clfs_all_pr_scores[k].append(v)    

        # for k,v in clfs_all_roc_scores.items():
        #     print('%s-roc:'%k,["%.3f"%i for i in v])
        
        # for k,v in clfs_all_pr_scores.items():
        #     print('%s-pr:'%k,["%.3f"%i for i in v])

        for k,v in clfs_all_roc_scores.items():
            print('%s avg roc_auc: %.2f, avg pr_auc: %.2f'%(k,np.mean(v),np.mean(clfs_all_pr_scores[k])))


        best_model = None
        best_avg_roc = float('-inf')
        best_avg_pr = float('-inf')
        
        for model, roc_score in clfs_all_roc_scores.items():
            avg_roc = sum(roc_score) / len(roc_score)
            avg_pr = sum(clfs_all_pr_scores[model]) / len(clfs_all_pr_scores[model])
            
            if avg_roc > best_avg_roc or (avg_roc == best_avg_roc and avg_pr > best_avg_pr):
                best_model = model
                best_avg_roc = avg_roc
                best_avg_pr = avg_pr

        print(f"The best model is {best_model} with an average ROC AUC score of {best_avg_roc:.2f} and an average precision-recall AUC score of {best_avg_pr:.2f}")

    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()

    print('\n\n')

