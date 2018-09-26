#
import pandas as pd

data = pd.read_csv('./dataset/data_fe.csv', dtype = {'fullVisitorId': 'str'})
test = pd.read_csv('./dataset/test_fe.csv', dtype = {'fullVisitorId': 'str'})
data_fe = pd.read_csv('./data_fe_olivier.csv')
test_fe = pd.read_csv('./test_fe_olivier.csv')

data_combine = pd.merge(data, data_fe, how='outer')
test_combine = pd.merge(test, test_fe, how='outer')
data_combine.drop(columns=['count_pageviews_per_network_domain'], inplace=True)
test_combine.drop(columns=['count_pageviews_per_network_domain'], inplace=True)
data_combine.to_csv('data_combine.csv', index=False)
test_combine.to_csv('test_combine.csv', index=False)
