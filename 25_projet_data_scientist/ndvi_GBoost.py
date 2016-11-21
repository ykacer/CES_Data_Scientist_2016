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

train = pd.read_csv('/home/kyoucef-smiths/Documents/boosting/train_modified.csv')
#target = 'Disbursed'
#IDcol = 'ID'
target = 'densite'
IDcol = 'name,population,surface'

def make_classification(x)
    if(x


def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors].values, dtrain[target]) 
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1] 
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=cv_folds, scoring='roc_auc')
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[target], dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
    #Print Feature Importances
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


predictors = [x for x in train.columns if 'cat' not in x and x not in IDcol.split(',') and x not in target.split(',')]

# first cv fitting
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, train, predictors)

# grid search : boosting parameters
param_test1 = {'n_estimators':range(20,81,10)}
estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)
gridsearch1 = GridSearchCV(estimator = estimator, param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gridsearch1.fit(train[predictors],train[target])

# grid search : tree parameters
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_features='sqrt', subsample=0.8, random_state=10)
gridsearch2 = GridSearchCV(estimator = estimator, param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gridsearch2.fit(train[predictors],train[target])
