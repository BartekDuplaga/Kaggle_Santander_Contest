#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:59:26 2019

@author: bartek
"""
import numpy as np # linear algebra
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('/Users/bartek/Desktop/Kaggle/Santander/Santander/train.csv')
test_df = pd.read_csv('/Users/bartek/Desktop/Kaggle/Santander/Santander/test.csv')

# data augementation
def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']

X_train = train_df[features]
y_train = target
X_tr, y_tr = augment(X_train.values, y_train.values)
print(X_tr.shape, y_tr.shape, y_tr.mean())

#podział na zbiory
X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
print(X_train.shape) 
print(X_test.shape) 
print(y_train.mean())
print(y_test.mean())

#badanie LGBM

parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 4,
    'max_bin': 100,
    'min_data_in_leaf': 400,
    'bagging_freq':1,
    'feature_fraction':0.5,
    'lambda_l2': 1  
}
lgb_train = lgb.Dataset(X_train, y_train)
lgb_model = lgb.train(parameters,lgb_train,num_boost_round=1000,verbose_eval=5)

y_hat = lgb_model.predict(X_test, num_iteration = lgb_model.best_iteration)
y_hat_df = pd.DataFrame(y_hat)
y_hat_df.hist(normed=True, cumulative= True, bins=100)
y_hat_df_1 = (y_hat_df>0.5).astype('int')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_hat_df_1)
print(cm)
a_score=accuracy_score(y_test,y_hat_df_1)
print(a_score)

#Krzywa ROC
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = lgb_model.predict(X_test, num_iteration = lgb_model.best_iteration)
fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
roc_auc = metrics.auc(fpr, tpr)
# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = lgb_model.predict(test_df.drop('ID_code',axis=1))
sub.to_csv('Kaggle_pred1.csv', index=False)

### LIGHT GBM

import lightgbm as lgb
params = {
    'task': 'train', 'max_depth': -1, 'boosting_type': 'gbdt',
    'objective': 'binary', 'num_leaves': 10, 'learning_rate': 0.005, 
    'max_bin': 500
}

y_hat = 0.0
for feature in features: # loop over all features
    lgb_train = lgb.Dataset(X_train[feature].values.reshape(-1,1), y_train.values)
    gbm = lgb.train(params, lgb_train, num_boost_round=500, verbose_eval=5)
    y_hat += gbm.predict(test_df[feature].values.reshape(-1,1), num_iteration=gbm.best_iteration)

   
y_hat /= len(features)
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = y_hat
sub.to_csv('Kaggle_pred1.csv', index=False)
"""
# Tunningowanie hiperparametrów
from sklearn.model_selection import GridSearchCV
param_grid = [
        {'min_data_in_leaf': [15,50,150],
         'num_leaves': [4,6,8],
         'learning_rate': [0.001, 0.01, 0.1]},
        {'min_data_in_leaf': [5,10],
         'num_leaves': [5,7],
         'learning_rate': [0.005, 0.05]}, ]
Booster = lgb.Booster(train_set=lgb_train) #lgb_train to instancja  lgb.Datasets
grid_search = GridSearchCV(Booster, param_grid, cv=5, scoring = 'roc_auc')
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score), params)

        




### LightGBM z Scikit Learn
