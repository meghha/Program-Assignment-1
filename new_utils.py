"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""

import numpy as np
import pandas as pd
from typing import Type, Dict
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_validate,
    KFold,
    ShuffleSplit
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
import utils as u
from sklearn.metrics import *

def scale_data(X):
        return (X - np.min(X)) / (np.max(X) - np.min(X))
    
def scale(X):
     min_value = 0
     max_value = 1
     if np.all((X >= min_value) & (X <= max_value)) & (X.dtype == "float64"):
            return True

def integerCheck_(y):
       if y.dtype=="int64":
              return True

def scores_(metrics):
  scores = {}
  scores['mean_fit_time'] = metrics['fit_time'].mean()
  scores['std_fit_time'] = metrics['fit_time'].std()
  scores['mean_accuracy'] = metrics['test_score'].mean()
  scores['std_accuracy'] = metrics['test_score'].std()
  return scores

def gridSearch_(clf,cv,Xtrain,ytrain):
    param_grid = {
    'n_estimators': [100, 300],  # Number of trees in the forest
    'max_depth': [10,20],  # Maximum depth of the tree
    'min_samples_split': [5,10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [10,20],  # Minimum number of samples required to be at a leaf node
    'criterion': ['gini','entropy']
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy',n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)
    mean_accuracy_cv = grid_search.cv_results_["mean_test_score"].mean()
    return grid_search, grid_search.best_estimator_, mean_accuracy_cv

def eda(Xtrain,Xtest,ytrain,ytest):
      nb_classes_train = len(set(ytrain))
      nb_classes_test = len(set(ytest))
      class_count_train = pd.DataFrame(ytrain).value_counts()
      class_count_test = pd.DataFrame(ytest).value_counts()
      length_Xtrain = Xtrain.shape[0]# Number of samples
      length_Xtest = Xtest.shape[0]
      length_ytrain = len(ytrain)
      length_ytest = len(ytest)
      max_Xtrain = np.max(Xtrain)
      max_Xtest = np.max(Xtest)
      return nb_classes_train,nb_classes_test,class_count_train,class_count_test,length_Xtrain,length_Xtest,length_ytrain,length_ytest,max_Xtrain,max_Xtest

def part1_partC(random_state,Xtrain,ytrain):
  answer = {}
  clf = DecisionTreeClassifier(random_state=random_state)
  cv = KFold(n_splits=5)
  metrics = u.train_simple_classifier_with_cv(Xtrain=Xtrain,ytrain=ytrain,clf=clf,cv=cv)
  scores = scores_(metrics=metrics)
  answer["scores"] = scores
  answer["clf"] = clf  # the estimator (classifier instance)
  answer["cv"] =  cv # the cross validator instance
  
  # the dictionary with the scores  (a dictionary with
  # keys: 'mean_fit_time', 'std_fit_time', 'mean_accuracy', 'std_accuracy'.
  
  return answer

def part1_partD(random_state,Xtrain,ytrain):
  answer = {}
  clf = DecisionTreeClassifier(random_state=random_state)
  cv = ShuffleSplit(random_state=random_state)
  metrics = u.train_simple_classifier_with_cv(Xtrain=Xtrain,ytrain=ytrain,clf=clf,cv=cv)
  # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'
  
  scores = scores_(metrics=metrics)

  answer = {}
  answer["scores"] = scores
  answer["clf"] = clf
  answer["cv"] = cv
  
  return answer

def part1_partF(random_state,Xtrain,ytrain,Xtest,ytest):
  clf_LR = LogisticRegression(max_iter=300,random_state=random_state,solver="saga")
  # clf_DT = DecisionTreeClassifier(random_state=random_state)

  cv = ShuffleSplit(random_state=random_state)

  # metrics_DT = u.train_simple_classifier_with_cv(Xtrain=Xtrain,ytrain=ytrain,clf=clf_DT,cv=cv)
  metrics_LR = u.train_simple_classifier_with_cv(Xtrain=Xtrain,ytrain=ytrain,clf=clf_LR,cv=cv)
  

  clf_LR.fit(Xtrain,ytrain)
  answer = {}
  # Enter your code, construct the `answer` dictionary, and return it.

  answer["clf"] = clf_LR
  # answer["clf_DT"] = clf_DT
  answer["cv"] = cv

  ytrain_pred = clf_LR.predict(Xtrain)
  ytest_pred = clf_LR.predict(Xtest)

  answer["scores_train_F"] = clf_LR.score(Xtrain,ytrain)
  answer["scores_test_F"] = clf_LR.score(Xtest,ytest)
  answer["mean_cv_accuracy"] = scores_(metrics_LR)["mean_accuracy"]
  answer["conf_mat_train"] = confusion_matrix(ytrain,ytrain_pred)
  answer["conf_mat_test"] = confusion_matrix(ytest,ytest_pred)

  # answer["scores_DT"] = scores_(metrics_DT)
  # answer["model_highest_accuracy"] = np.where(answer["scores_RF"]["mean_accuracy"]>answer["scores_DT"]["mean_accuracy"],"Random-Forest","Decision-Tree")
  # answer["model_lowest_variance"] = np.where(answer["scores_RF"]["std_accuracy"]>answer["scores_DT"]["std_accuracy"],"Random-Forest","Decision-Tree")
  # answer["model_fastest"] = np.where(answer["scores_RF"]["mean_fit_time"]>answer["scores_DT"]["mean_fit_time"],"Random-Forest","Decision-Tree")

  return answer

def value_counts(y):
     unique_values, counts = np.unique(y, return_counts=True)
     return list(counts)


def filter_out_7_9s(X: NDArray[np.floating], y: NDArray[np.int32]):
    """
    Filter the dataset to include only the digits 7 and 9, and return an imbalanced dataset with 90% of 9s removed
    Parameters:
        X: Data matrix
        y: Labels
    Returns:
        Filtered data matrix and labels
    Notes:
        np.int32 is a type with a range based on 32-bit ints
        np.int has no bound; it can hold arbitrarily long numbers
    """
    seven_nine_idx = (y == 7) | (y == 9)
    X_binary = X[seven_nine_idx, :]
    y_binary = y[seven_nine_idx]

    nine_indices = np.where(y_binary== 9)[0]
    num_to_remove = int(0.9 * len(nine_indices))
    selected_nine_indices = np.random.choice(nine_indices, size=num_to_remove, replace=False)
    X_binary = np.delete(X_binary, selected_nine_indices,axis=0)
    y_binary = np.delete(y_binary, selected_nine_indices)

    y_binary = np.where(y_binary==7,0,1)
    
    return X_binary, y_binary

def replace_7_9s(X: NDArray[np.floating], y: NDArray[np.int32]):
  seven_nine_idx = (y == 7) | (y == 9)
  X_binary = X[seven_nine_idx, :]
  y_binary = y[seven_nine_idx]
  y_binary = np.where(y_binary==7,0,1)
  return X_binary,y_binary

def scores_part3(metrics):
  scores = {}
  scores["mean_accuracy"] = metrics['accuracy'].mean()
  scores["mean_recall"] = metrics['recall'].mean()
  scores["mean_precision"] = metrics['precision'].mean()
  scores["mean_F1"] = metrics['F1'].mean()
  scores["std_accuracy"] = metrics['accuracy'].std()
  scores["std_recall"] = metrics['recall'].std()
  scores["std_precision"] = metrics['precision'].std()
  scores["std_F1"] = metrics['F1'].std()
  return scores

def calculate_top_k_scores(clf, Xtrain, ytrain, Xtest, ytest, k_values):
    train_scores = []
    test_scores = []

    for k in k_values:
        train_score = top_k_accuracy_score(ytrain, clf.predict_proba(Xtrain), k=k)
        test_score = top_k_accuracy_score(ytest, clf.predict_proba(Xtest), k=k)
        train_scores.append((k, train_score))
        test_scores.append((k, test_score))

    return train_scores, test_scores

def plot_k_vs_score(scores, title):
    k_values, accuracy_scores = zip(*scores)
    plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('k')
    plt.ylabel('Top k Accuracy Score')
    plt.grid(True)
    plt.show()




       
       


