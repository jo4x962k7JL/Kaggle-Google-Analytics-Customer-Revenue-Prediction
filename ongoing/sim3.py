#
import gc, time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMRegressor
start = time.time()
# RANDOM = 47

data = pd.read_csv('./data_combine.csv', dtype = {'fullVisitorId': 'str'})
test = pd.read_csv('./test_combine.csv', dtype = {'fullVisitorId': 'str'})

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
lgb_params = {'learning_rate': 0.03, 'n_estimators': 2000, 'num_leaves': 128, 'subsample': 0.2217,
              'colsample_bytree': 0.6810, 'min_split_gain': np.power(10.0, -4.9380), 'reg_alpha': np.power(10.0, -3.2454),
              'reg_lambda': np.power(10.0, -4.8571), 'min_child_weight': np.power(10.0, 2), 'silent': True}

oof_pred = np.zeros(X_train.shape[0])
y_pred = np.zeros(X_test.shape[0])
folds = StratifiedKFold(n_splits= 5, shuffle=True)
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
    train_x, train_y = X_train.iloc[train_idx], y_train.iloc[train_idx]
    valid_x, valid_y = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

    reg = LGBMRegressor(**lgb_params)
    reg.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric= 'rmse', verbose= 1000, early_stopping_rounds= 100)

    oof_pred[valid_idx] = reg.predict(valid_x)
    y_pred_ = reg.predict(X_test) / folds.n_splits
    y_pred_[y_pred_ < 0] = 0
    y_pred += y_pred_
    oof_pred[oof_pred < 0] = 0
    print('Fold {:d} RMSE : {:.5f}'.format(n_fold + 1, mean_squared_error(valid_y, oof_pred[valid_idx])**0.5))
print('Full RMSE: {:.5f}'.format(mean_squared_error(y_train, oof_pred)**0.5))

csv_lgbm['PredictedLogRevenue'] = y_pred
csv_lgbm['PredictedLogRevenue'] = csv_lgbm['PredictedLogRevenue'].apply(lambda x : 0.0 if x < 0 else x)
csv_lgbm['PredictedLogRevenue'].fillna(0.0, inplace=True)
csv_lgbm = csv_lgbm.groupby('fullVisitorId').sum().reset_index()
csv_lgbm.to_csv('csv_sim3.csv', index = False)
print('Run time: {:.2f}mins'.format((time.time() - start)/60))

oof_lgbm['PRED'] = oof_pred
oof_lgbm.to_csv('oof_sim3.csv', index = False)
