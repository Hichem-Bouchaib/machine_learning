import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
import urllib.request
from sklearn import datasets, svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
#from sklearn.grid_search import GridSearchCV

print("Beginning...")
dataset = np.loadtxt("data.txt", delimiter=",")

X_train=dataset[0:920,0:19]
y_train=dataset[0:920,19:]
X_test=dataset[920:,0:19]
y_test=dataset[920:,19:]

########################## GridSearchCV commented to gain time on script's running ##############
#parameter_candidates = [
 # {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  #{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
   # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly']},
#]

#clf4 = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
#clf4.fit(X_test,y_test.ravel())


# View the accuracy score
#print('Best score for X:', clf4.best_score_) 

# View the best parameters for the model found using grid search
#print('Best C:',clf4.best_estimator_.C) 
#print('Best Kernel:',clf4.best_estimator_.kernel)
#print('Best Gamma:',clf4.best_estimator_.gamma)

######################### End of GridSearchCVs section #######################

#Linear
clf = svm.SVC(kernel="linear",C=10)
grid_clf=clf.fit(X_train,y_train.ravel())
print('LINEAR training score is %s' % clf.score(X_train, y_train))
print('LINEAR test score is %s' % clf.score(X_test, y_test))

#Poly
clf2 = svm.SVC(kernel="poly", gamma=0.0001)
clf2.fit(X_train,y_train.ravel())
print('POLY training score is %s' % clf2.score(X_train, y_train))
print('POLY test score is %s' % clf2.score(X_test, y_test))

#RBF
clf3 = svm.SVC(kernel="rbf",gamma=0.0001)
clf3.fit(X_train,y_train.ravel())
print('RBF training score is %s ' % clf3.score(X_train, y_train))
print('RBF test score is %s' % clf3.score(X_test, y_test))


clf5 = RandomForestClassifier(n_estimators=10)
clf5.fit(X_train, y_train.ravel())
print("Training set with Random forest : %s" % clf5.score(X_train,y_train))
print("Testing set with Random forest : %s" % clf5.score(X_test,y_test))

