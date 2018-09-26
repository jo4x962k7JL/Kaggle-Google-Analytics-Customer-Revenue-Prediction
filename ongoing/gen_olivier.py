#
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import functools
from multiprocessing import Pool
import gc, sys

def get_dict_content(dct_str=None, key='visits'):
    try:
        return float(eval(dct_str)[key])
    except KeyError:
        return 0.0

def get_dict_content_str(dct_str=None, key='visits'):
    try:
        return eval(dct_str)[key]
    except NameError:
        return eval(dct_str.replace('false', 'False').replace('true', 'True'))[key]
    except KeyError:
        return np.nan

def apply_func_on_series(data=None, func=None, key=None):
    return data.apply(lambda x: func(x, key=key))

def multi_apply_func_on_series(df=None, func=None, key=None, n_jobs=4):
    p = Pool(n_jobs)
    f_ = p.map(functools.partial(apply_func_on_series, func=func, key=key), np.array_split(df, n_jobs))
    f_ = pd.concat(f_, axis=0, ignore_index=True)
    p.close()
    p.join()
    return f_.values

def read_file(file_name=None):
    return pd.read_csv(file_name,
        usecols=['channelGrouping', 'date', 'fullVisitorId', 'sessionId', 'totals', 'device', 'geoNetwork', 'socialEngagementType', 'trafficSource'],
        dtype={'channelGrouping': str, 'geoNetwork': str, 'date': str, 'fullVisitorId': str, 'sessionId': str, 'totals': str, 'device': str})

def populate_data(df=None):
    df['date'] = pd.to_datetime(df['date'])
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_doy'] = df['date'].dt.dayofyear
    df['sess_date_mon'] = df['date'].dt.month
    df['sess_date_week'] = df['date'].dt.weekofyear

    for f in ['transactionRevenue', 'visits', 'hits', 'pageviews', 'bounces', 'newVisits']:
        df[f] = multi_apply_func_on_series(df=df['totals'], func=get_dict_content, key=f, n_jobs=4)
    for f in ['continent', 'subContinent', 'country', 'region', 'metro', 'city', 'networkDomain']:
        df[f] = multi_apply_func_on_series(df=df['geoNetwork'], func=get_dict_content_str, key=f, n_jobs=4)
    for f in ['browser', 'operatingSystem', 'isMobile', 'deviceCategory']:
        df[f] = multi_apply_func_on_series(df=df['device'], func=get_dict_content_str, key=f, n_jobs=4)
    for f in ['source', 'medium']:
        df[f] = multi_apply_func_on_series(df=df['trafficSource'], func=get_dict_content_str, key=f, n_jobs=4)

    df.drop(['totals', 'geoNetwork', 'device', 'trafficSource'], axis=1, inplace=True)
    df['dummy'] = 1
    df['user_cumcnt_per_day'] = (df[['fullVisitorId','date', 'dummy']].groupby(['fullVisitorId','date'])['dummy'].cumcount()+1)
    df['user_sum_per_day'] = df[['fullVisitorId','date', 'dummy']].groupby(['fullVisitorId','date'])['dummy'].transform(sum)
    df['user_cumcnt_sum_ratio_per_day'] = df['user_cumcnt_per_day'] / df['user_sum_per_day']
    df.drop('dummy', axis=1, inplace=True)
    return df

def factorize_categoricals(df=None, cat_indexers=None):
    cat_feats = [f for f in df.columns if ((df[f].dtype == 'object') & (f not in ['fullVisitorId', 'sessionId', 'date', 'totals', 'device', 'geoNetwork', 'device', 'trafficSource']))]
    if cat_indexers is None:
        cat_indexers = {}
        for f in cat_feats:
            df[f], indexer = pd.factorize(df[f])
            cat_indexers[f] = indexer
    else:
        for f in cat_feats:
            df[f] = cat_indexers[f].get_indexer(df[f])
    return df, cat_indexers, cat_feats

def aggregate_sessions(df=None, cat_feats=None, sum_of_logs=False):
    if sum_of_logs is True:
        df['transactionRevenue'] = np.log1p(df['transactionRevenue'])
    aggs = {'date': ['min', 'max'], 'transactionRevenue': ['sum', 'size'], 'hits': ['sum', 'min', 'max', 'mean', 'median'],
        'pageviews': ['sum', 'min', 'max', 'mean', 'median'], 'bounces': ['sum', 'mean', 'median'], 'newVisits': ['sum', 'mean', 'median']}
    for f in cat_feats + ['sess_date_dow', 'sess_date_doy', 'sess_date_mon', 'sess_date_week']:
        aggs[f] = ['min', 'max', 'mean', 'median', 'var', 'std']

    users = df.groupby('fullVisitorId').agg(aggs).reset_index()
    users.columns = pd.Index(['fullVisitorId'] + [k + '_' + agg for k in aggs.keys() for agg in aggs[k]])

    users['date_diff'] = (users.date_max - users.date_min).astype(np.int64) // (24 * 3600 * 1e9)
    if sum_of_logs is False:
        users['transactionRevenue_sum'] = np.log1p(users['transactionRevenue_sum'])
    return users

data = read_file('./dataset/train.csv')
test = read_file('./dataset/test.csv')
data = populate_data(df=data)
test = populate_data(df=test)

data, cat_indexers, cat_feats = factorize_categoricals(df=data)
data = aggregate_sessions(df=data, cat_feats=cat_feats)
data.drop(['date_min', 'date_max', 'transactionRevenue_sum'], axis=1, inplace=True)
test, cat_indexers, cat_feats = factorize_categoricals(df=test, cat_indexers=cat_indexers)
test = aggregate_sessions(df=test, cat_feats=cat_feats)
test.drop(['date_min', 'date_max', 'transactionRevenue_sum'], axis=1, inplace=True)

data.to_csv('data_fe_olivier.csv', index=False)
test.to_csv('test_fe_olivier.csv', index=False)

# Reference
# https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480/code
