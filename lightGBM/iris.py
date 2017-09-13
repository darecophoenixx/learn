# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import log_loss

import lightgbm as lgb


iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target
N = Y.size

'''
トレーニングデータとバリデーションデータに分割する
'''
index = np.arange(N)
xtrain = X[index[index % 2 != 0],:]
ytrain = Y[index[index % 2 != 0]]

xtest = X[index[index % 2 == 0],:]
yans = Y[index[index % 2 == 0]]

lgb_train = lgb.Dataset(xtrain, label=ytrain)
lgb_eval = lgb.Dataset(xtest, label=yans)

params = {
    'task': 'train',
    'num_class': 3,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'verbose': 0
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval)

print('Feature names:', gbm.feature_name())

'''
利用した変数の重要度をみる
'''
print('Feature importances:', list(gbm.feature_importance()))
lgb.plot_importance(gbm, max_num_features=10)
plt.show()

'''
トレーニングデータにおいて、推定する
'''
y_pred = gbm.predict(xtrain, num_iteration=gbm.best_iteration)
print(y_pred)
print(ytrain)
print('The log_loss of prediction is:', log_loss(ytrain, y_pred))

'''
バリデーションデータにおいて、推定する
'''
y_pred = gbm.predict(xtest, num_iteration=gbm.best_iteration)
print(y_pred)
print(yans)
print('The log_loss of prediction is:', log_loss(yans, y_pred))



