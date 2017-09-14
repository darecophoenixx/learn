# -*- encoding: utf-8 -*-
'''
irisデータをpd.DataFrameで作成する
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import train_test_split

import lightgbm as lgb

'''
irisデータをpandasのDataFrameでインスタンス化します
'''
iris = datasets.load_iris()
iris2 = pd.DataFrame(iris.data, columns=iris.feature_names)
'''種をcls列にセットします'''
iris2['cls'] = pd.Categorical(iris.target)

N = iris2.shape[0]
index = np.arange(N)
x_train0 = index[index % 2 != 0]
x_test0 = index[index % 2 == 0]

x_train = iris2.loc[x_train0]
x_test = iris2.loc[x_test0]

'''
lightGBM用のデータセットを作ります
説明変数側はcls列は必要ないので削除します
'''
lgb_train = lgb.Dataset(x_train.drop(['cls'], axis=1), label=x_train.cls)
lgb_eval = lgb.Dataset(x_test.drop(['cls'], axis=1), label=x_test.cls)

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
print('Feature importances:', list(gbm.feature_importance()))
ax = lgb.plot_importance(gbm, max_num_features=10)
plt.show()

