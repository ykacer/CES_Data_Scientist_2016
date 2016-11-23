#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('France/ndvi_features.csv')
#target = 'Disbursed'
#IDcol = 'ID'
target = u'label'
IDcol = u'name,population,surface,densite'

density = train['population'].as_matrix()/train['surface'].as_matrix();
train[target] = 0
label = -1*np.ones(train.shape[0])
for i,n in enumerate(density):
    if n>10000:
        label[i] = 0
    elif n>5000:
        label[i] = 1
    elif n>500:
        label[i] = 2
    else:
        label[i] = 3

train[target] = label


def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors].values, dtrain[target]) 
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1] 
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=cv_folds)
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[target], dtrain_predictions)
    print "Confusion Matrix (Train): \n"+np.array_str(metrics.confusion_matrix(dtrain[target].as_matrix(), dtrain_predictions))
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
    #Print Feature Importances
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
	plt.show()


predictors = [x for x in train.columns if 'cat' not in x and x not in IDcol.split(',') and x not in target.split(',')]

# first cv fitting
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, train, predictors)

# grid search : boosting parameters
param_test1 = {'n_estimators':range(20,81,10)}
estimator = GradientBoostingClassifier(learning_rate=0.1,max_depth=17,min_samples_split=30,min_samples_leaf=30,max_features='sqrt',subsample=0.8,random_state=10)
gridsearch1 = GridSearchCV(estimator = estimator, param_grid = param_test1,n_jobs=4,iid=False, cv=5)
gridsearch1.fit(train[predictors],train[target])

# grid search : tree parameters
param_test2 = {'max_depth':range(7,25,2), 'min_samples_split':range(100,201,10)}
estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_features='sqrt', subsample=0.8, random_state=10)
gridsearch2 = GridSearchCV(estimator = estimator, param_grid = param_test2,n_jobs=4,iid=False, cv=5)
gridsearch2.fit(train[predictors],train[target])

# grid search : tree parameters
param_test3 = {'min_samples_split':range(190,300,20), 'min_samples_leaf':range(30,71,10)}
estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70,max_depth=21,max_features='sqrt', subsample=0.8, random_state=10)
gridsearch3 = GridSearchCV(estimator = estimator, param_grid = param_test3,n_jobs=4,iid=False, cv=5)
gridsearch3.fit(train[predictors],train[target])

# grid search : max features
param_test4 = {'max_features':range(7,22,2)}
estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=21, min_samples_split=210, min_samples_leaf=60, subsample=0.8, random_state=10)
gridsearch4 = GridSearchCV(estimator = estimator, param_grid = param_test4,n_jobs=4,iid=False, cv=5)
gridsearch4.fit(train[predictors],train[target])

# grid search : subsample
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=21, min_samples_split=210, min_samples_leaf=60, max_features=15, random_state=10)
gsearch5 = GridSearchCV(estimator = estimator, param_grid = param_test5,n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])

# grid search : learning_rate
param_test6 = {'learning_rate':[0.01]}
estimator = GradientBoostingClassifier(n_estimators=700, max_depth=21, min_samples_split=210, min_samples_leaf=60, max_features=15, subsample=0.8, random_state=10)
gsearch6 = GridSearchCV(estimator = estimator, param_grid = param_test6,n_jobs=4,iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])
