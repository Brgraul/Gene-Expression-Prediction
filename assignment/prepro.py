# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 21:36:11 2017

@author: tirabe
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

# File that contains several methods to preprocess the data
def pca(x_train, x_test):
    normalizer = Normalizer()
    normalizer.fit(x_train)
    x_train = normalizer.transform(x_train)
    x_test = normalizer.transform(x_test)
    
    pca = PCA()
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
