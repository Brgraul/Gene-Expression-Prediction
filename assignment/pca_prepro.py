# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 21:24:38 2017

@author: tirabe
"""

C_range = 10.0 ** np.arange(-4, 3)

grid = {'penalty': ['l1', 'l2'], 'C': C_range}
clf = GridSearchCV(KNeighborsClassifier(), grid, cv=5, scoring = 'roc_auc')
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

 print("Starting KNN")
 KNN = sklearn.neighbors.KNeighborsClassifier(algorithm = "ball_tree")
 KNN.fit(X = x_train, y = y_train)
 #res = KNN.predict(x_test)
 scores = cross_val_score(KNN, x_train, y_train, scoring="roc_auc")
 print ("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

 
 