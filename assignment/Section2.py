# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 21:07:54 2017

@author: tirabe
"""

# Section 2 of the assignment
import numpy as n
from sklearn.linear_model import LogisticRegression


clf = LogisticRegression()
clf.fit(x_train, y_train)
y_prob = clf.predict_proba(x_test)[:,1].ravel()
generate_submission(y_prob)