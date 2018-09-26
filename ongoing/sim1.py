#
import gc, time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
from math import sqrt
start = time.time()
RANDOM = 47

data = pd.read_csv('./dataset/data_fe.csv', dtype = {'fullVisitorId': 'str'})
test = pd.read_csv('./dataset/test_fe.csv', dtype = {'fullVisitorId': 'str'})

# X_train = data.drop(['fullVisitorId', 'totals_transactionRevenue', 'trafficSource_adwordsClickInfo.page'], axis=1)
X_train = data.drop(['fullVisitorId', 'totals_transactionRevenue'], axis=1)
y_train = data['totals_transactionRevenue']
# X_test = test.drop(['fullVisitorId', 'totals_transactionRevenue', 'trafficSource_adwordsClickInfo.page'], axis=1)
X_test = test.drop(['fullVisitorId', 'totals_transactionRevenue'], axis=1)

csv_lgbm = test[['fullVisitorId']].copy()
oof_lgbm = data[['fullVisitorId', 'totals_transactionRevenue']].copy()

print('X_train size: {}\nX_test size: {}'.format(X_train.shape, X_test.shape))
del data, test
gc.collect()

print('======================================== Start training LGBM ========================================')
lgb_params = {"objective" : "regression", "metric" : "rmse", "max_depth": 8, "min_child_samples": 20,
               "reg_alpha": 1, "reg_lambda": 1, "num_leaves" : 257, "learning_rate" : 0.01,
               "subsample" : 0.8, "colsample_bytree" : 0.8, "verbosity": -1}

oof_pred = np.zeros(X_train.shape[0])
y_pred = np.zeros(X_test.shape[0])
folds = KFold(n_splits= 5, shuffle=True, random_state=RANDOM)
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
    valid_x, valid_y = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
    train_data = lgb.Dataset(X_train.iloc[train_idx], label=y_train.iloc[train_idx])
    valid_data = lgb.Dataset(X_train.iloc[valid_idx], label=y_train.iloc[valid_idx])
    num_boost_round = 20000
    clf = lgb.train(lgb_params, train_data, num_boost_round, valid_sets = [train_data, valid_data], verbose_eval=1000, early_stopping_rounds = 100)
    oof_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
    y_pred += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
    print('Fold {:2d} RMSE : {:.6f}'.format(n_fold + 1, sqrt(mean_squared_error(valid_y, oof_pred[valid_idx]))))
print('Full RMSE {:.6f}'.format(sqrt(mean_squared_error(y_train, oof_pred))))

csv_lgbm['PredictedLogRevenue'] = y_pred
csv_lgbm['PredictedLogRevenue'] = csv_lgbm['PredictedLogRevenue'].apply(lambda x : 0.0 if x < 0 else x)
csv_lgbm['PredictedLogRevenue'].fillna(0.0, inplace=True)
csv_lgbm = csv_lgbm.groupby('fullVisitorId').sum().reset_index()
csv_lgbm.to_csv('csv_sim1.csv', index = False)
print('Run time: {:.2f}mins'.format((time.time() - start)/60))

oof_lgbm['PRED'] = oof_pred
oof_lgbm.to_csv('oof_sim1.csv', index = False)


# X_train size: (903653, 219)
# X_test size: (804684, 219)
# ======================================== Start training LGBM ========================================
# Training until validation scores don't improve for 100 rounds.
# Early stopping, best iteration is:
# [744]	training's rmse: 0.715798	valid_1's rmse: 0.770845
# Fold  1 RMSE : 0.770845
# Training until validation scores don't improve for 100 rounds.
# Early stopping, best iteration is:
# [725]	training's rmse: 0.717665	valid_1's rmse: 0.772591
# Fold  2 RMSE : 0.772591
# Training until validation scores don't improve for 100 rounds.
# [1000]	training's rmse: 0.70712	valid_1's rmse: 0.775179
# Early stopping, best iteration is:
# [956]	training's rmse: 0.708514	valid_1's rmse: 0.775093
# Fold  3 RMSE : 0.775093
# Training until validation scores don't improve for 100 rounds.
# Early stopping, best iteration is:
# [752]	training's rmse: 0.712161	valid_1's rmse: 0.79068
# Fold  4 RMSE : 0.790680
# Training until validation scores don't improve for 100 rounds.
# Early stopping, best iteration is:
# [612]	training's rmse: 0.725064	valid_1's rmse: 0.768148
# Fold  5 RMSE : 0.768148
# Full RMSE 0.775512
# Run time: 8.81mins
