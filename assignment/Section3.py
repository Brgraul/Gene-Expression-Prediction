# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 21:09:48 2017

@author: tirabe
"""

# Section 3 & 4 of the assignment
C_range = 10.0 ** np.arange(-4, 3)

grid = {'penalty': ['l1', 'l2'], 'C': C_range}
clf = GridSearchCV(LogisticRegression(), grid, cv=5, scoring = 'roc_auc')
clf.fit(x_train, y_train)

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
