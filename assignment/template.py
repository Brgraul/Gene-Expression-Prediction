# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 17:07:19 2017

@author: lingyu, hehu
"""

# Basic Modules
import os
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import train_test_split

# Hyperparameter optimization

from sklearn.grid_search import GridSearchCV

# Mathematical models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Accuracy evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score

# Output 
from generate_submission import generate_submission



# Data loading module

if __name__== '__main__':

    data_path = "data" # This folder holds the csv files

    # load csv files. We use np.loadtxt. Delimiter is ","
    # and the text-only header row will be skipped.
    
    print("Loading data...")
    x_train = np.loadtxt(data_path + os.sep + "x_train.csv", 
                         delimiter = ",", skiprows = 1)
    x_test  = np.loadtxt(data_path + os.sep + "x_test.csv", 
                         delimiter = ",", skiprows = 1)    
    y_train = np.loadtxt(data_path + os.sep + "y_train.csv", 
                         delimiter = ",", skiprows = 1)
    
    print "All files loaded. Preprocessing..."

    # remove the first column(Id)
    x_train = x_train[:,1:] 
    x_test  = x_test[:,1:]   
    y_train = y_train[:,1:] 

    # Every 100 rows correspond to one gene.
    # Extract all 100-row-blocks into a list using np.split.
    num_genes_train = x_train.shape[0] / 100
    num_genes_test  = x_test.shape[0] / 100

    print("Train / test data has %d / %d genes." % \
          (num_genes_train, num_genes_test))
    x_train = np.split(x_train, num_genes_train)
    x_test  = np.split(x_test, num_genes_test)

    # Reshape by raveling each 100x5 array into a 500-length vector
    x_train = [g.ravel() for g in x_train]
    x_test  = [g.ravel() for g in x_test]
    
    # convert data from list to array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test  = np.array(x_test)
    y_train = np.ravel(y_train)
    
    # Now x_train should be 15485 x 500 and x_test 3871 x 500.
    # y_train is 15485-long vector.
    
    print("x_train shape is %s" % str(x_train.shape))    
    print("y_train shape is %s" % str(y_train.shape))
    print("x_test shape is %s" % str(x_test.shape))
    
    print('Data preprocessing done...')
    
    print("Next steps FOR YOU:")
    print("-" * 30)
    print("1. Define a classifier using sklearn")
    print("2. Assess its accuracy using cross-validation (optional)")
    print("3. Fine tune the parameters and return to 2 until happy (optional)")
    print("4. Create submission file. Should be similar to y_train.csv.")
    print("5. Submit at kaggle.com and sit back.")
    
    
    
# Section 2 of the assignment

clf = LogisticRegression()
clf.fit(x_train, y_train)
y_prob = clf.predict_proba(x_test)[:,1].ravel()
generate_submission(y_prob)



# Section 3 of the assignment

C_range = 10.0 ** np.arange(-4, 3)

grid = {'penalty': ['l1', 'l2'], 'C': C_range}
clf = GridSearchCV(LogisticRegression(), grid, cv=5, scoring = 'roc_auc')
clf.fit(x_train, y_train)



# Section 4 of the assignment 

normalizer = Normalizer()
normalizer.fit(x_train)
x_train = normalizer.transform(x_train)
x_test = normalizer.transform(x_test)
pca = PCA()
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
accuracies = []
for num_components in range(1, 500):
    clf = LogisticRegression(C=10.0 ** -1, penalty="l1")
    clf.fit(x_train[:, :num_components], y_train)
    scores = cross_val_score(clf, x_train, y_train, cv = 5, n_jobs = 2)
    accuracies.append(np.mean(scores))    
    
plt.plot(range(1,500), accuracies)



# Section 5 of the assignment 


# Gridsearch and PCA + KNN stack

# Optimization of the parameters bia Gridsearch with Cross Validation
n_neighbors = np.arange(1, 100)

grid = {'n_neighbors': n_neighbors, 'weights': ['uniform', 'distance'], 'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}
clf = GridSearchCV(LogisticRegression(), grid, cv=5, scoring = 'roc_auc')
clf.fit(x_train, y_train)

# Optimization of the PCA 

normalizer = Normalizer()
normalizer.fit(x_train)
x_train = normalizer.transform(x_train)
x_test = normalizer.transform(x_test)
pca = PCA()
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
accuracies = []

for num_components in range(1, 500):
    clf = KNeighborsClassifier()
    clf.fit(x_train[:, :num_components], y_train)
    scores = cross_val_score(clf, x_train, y_train, cv = 5, n_jobs = 2)
    accuracies.append(np.mean(scores))    
    
plt.plot(range(1,500), accuracies)



# Gridsearch and PCA + SVM stack

# Optimization of the parameters bia Gridsearch with Cross Validation

C_range = 10.0 ** np.arange(-4, 3)

grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'C': C_range}
clf = GridSearchCV(LogisticRegression(), grid, cv=5, scoring = 'roc_auc')
clf.fit(x_train, y_train)

# Optimization of the PCA 

normalizer = Normalizer()
normalizer.fit(x_train)
x_train = normalizer.transform(x_train)
x_test = normalizer.transform(x_test)
pca = PCA()
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
accuracies = []

for num_components in range(1, 500):
    clf = SVC()
    clf.fit(x_train[:, :num_components], y_train)
    scores = cross_val_score(clf, x_train, y_train, cv = 5, n_jobs = 2)
    accuracies.append(np.mean(scores))    
    
plt.plot(range(1,500), accuracies)

