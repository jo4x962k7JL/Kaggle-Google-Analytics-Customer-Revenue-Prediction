#
import os,json, gc, time
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.preprocessing import LabelEncoder
start = time.time()

# Load dataset with json
def load_df_with_json(path, json_cols, dtype_dict):
    df = pd.read_csv(path, dtype = dtype_dict, converters={column: json.loads for column in json_cols}, parse_dates=['date'])
    for col in json_cols:
        json_df = json_normalize(df[col])
        json_df.columns = [f"{col}_{sub_col}" for sub_col in json_df.columns]
        df = df.drop(col, axis=1).merge(json_df, right_index=True, left_index=True)
    return df

json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
dtype_dict = {'fullVisitorId': 'str'}
data = load_df_with_json(path='./dataset/train.csv', json_cols=json_cols, dtype_dict=dtype_dict)
test = load_df_with_json(path='./dataset/test.csv', json_cols=json_cols, dtype_dict=dtype_dict)

# Drop columns with just one value or all unknown
# do not use nunique(dropna=False). The different cols includes:
# {'totals_bounces', 'totals_newVisits', 'trafficSource_adwordsClickInfo.isVideoAd', 'trafficSource_campaignCode', 'trafficSource_isTrueDirect'}
drop_cols = [col for col in data.columns if data[col].nunique() == 1]
data.drop(drop_cols, axis = 1, inplace=True)
test.drop([col for col in drop_cols if col in test.columns], axis = 1, inplace=True)

# Fill transactionRevenue with zeroes and convert its type to numeric
data['totals_transactionRevenue'] = data['totals_transactionRevenue'].astype(float)
data['totals_transactionRevenue'].fillna(0.0, inplace=True)
data['totals_transactionRevenue'] = np.log1p(data['totals_transactionRevenue'])

def process_date_time(data_df):
    data_df['year'] = data_df['date'].dt.year
    data_df['month'] = data_df['date'].dt.month
    data_df['day'] = data_df['date'].dt.day
    data_df['weekday'] = data_df['date'].dt.weekday
    data_df['weekofyear'] = data_df['date'].dt.weekofyear
    data_df['month_unique_user_count'] = data_df.groupby('month')['fullVisitorId'].transform('nunique')
    data_df['day_unique_user_count'] = data_df.groupby('day')['fullVisitorId'].transform('nunique')
    data_df['weekday_unique_user_count'] = data_df.groupby('weekday')['fullVisitorId'].transform('nunique')
    return data_df

def process_format(data_df):
    for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
        data_df[col] = data_df[col].astype(float)
    return data_df

def process_device(data_df):
    data_df['browser_category'] = data_df['device_browser'] + '_' + data_df['device_deviceCategory']
    data_df['browser_os'] = data_df['device_browser'] + '_' + data_df['device_operatingSystem']
    return data_df

def process_totals(data_df):
    data_df['visitNumber'] = np.log1p(data_df['visitNumber'])
    data_df['totals_hits'] = np.log1p(data_df['totals_hits'])
    data_df['totals_pageviews'] = np.log1p(data_df['totals_pageviews'].fillna(0))
    data_df['mean_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('mean')
    data_df['sum_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('sum')
    data_df['max_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('max')
    data_df['min_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('min')
    data_df['var_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('var')
    data_df['mean_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('mean')
    data_df['sum_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('sum')
    data_df['max_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('max')
    data_df['min_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('min')
    return data_df

def process_geo_network(data_df):
    data_df['sum_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    data_df['count_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('count')
    data_df['mean_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('mean')
    data_df['sum_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    data_df['count_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    data_df['mean_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('mean')
    return data_df

def process_traffic_source(data_df):
    data_df['source_country'] = data_df['trafficSource_source'] + '_' + data_df['geoNetwork_country']
    data_df['campaign_medium'] = data_df['trafficSource_campaign'] + '_' + data_df['trafficSource_medium']
    data_df['medium_hits_mean'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('mean')
    data_df['medium_hits_max'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('max')
    data_df['medium_hits_min'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('min')
    data_df['medium_hits_sum'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('sum')
    return data_df

data = process_date_time(data)
data = process_format(data)
data = process_device(data)
data = process_totals(data)
data = process_geo_network(data)
data = process_traffic_source(data)

test = process_date_time(test)
test = process_format(test)
test = process_device(test)
test = process_totals(test)
test = process_geo_network(test)
test = process_traffic_source(test)

data['device_isMobile'].replace({True: 1, False: 0}, inplace=True)
test['device_isMobile'].replace({True: 1, False: 0}, inplace=True)

data = data.sort_values(by=['fullVisitorId']).reset_index(drop=True)
test = test.sort_values(by=['fullVisitorId']).reset_index(drop=True)

train_size = data.shape[0]
merged_df = pd.concat([data, test], sort=True).reset_index(drop=True)
merged_df['trafficSource_adContent'].replace({'Google store': 'Google Store', 'google store': 'Google Store'}, inplace=True)

# One-Hot Encoding and Label Encoding
ohe_cols = [col for col in merged_df.columns if (merged_df[col].dtype == 'object') and
       (col not in ['fullVisitorId', 'sessionId', 'trafficSource_referralPath']) and (merged_df[col].nunique() < 50)]
label_cols = [col for col in merged_df.columns if (merged_df[col].dtype == 'object') and
       (col not in ['fullVisitorId', 'sessionId', 'trafficSource_referralPath']) and (merged_df[col].nunique() >= 50)]

merged_df = pd.get_dummies(merged_df, columns = ohe_cols, dummy_na=True)
for col in label_cols:
    merged_df[col] = LabelEncoder().fit_transform(merged_df[col].values.astype('str'))

merged_df.loc[:train_size-1, 'totals_transactionRevenue'] = \
        merged_df.loc[:train_size-1, 'totals_transactionRevenue'].apply(lambda y: np.digitize(y, bins=np.arange(2, 24, 2)))

merged_df.drop(['visitNumber', 'date', 'sessionId', 'visitId', 'visitStartTime', 'trafficSource_referralPath'], axis = 1, inplace=True)

data = merged_df[:train_size]
test = merged_df[train_size:]
data.to_csv('./dataset/data_fe.csv', index=False)
test.to_csv('./dataset/test_fe.csv', index=False)
print('Run time: {:.2f}mins'.format((time.time() - start)/60))

# Reference
# Deal with json
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields
# Drop outlier columns
# https://www.kaggle.com/ogakulov/feature-engineering-step-by-step
# Feature engineering
# https://www.kaggle.com/youhanlee/stratified-sampling-for-regression-lb-1-6595
