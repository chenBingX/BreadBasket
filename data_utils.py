# coding=utf-8

import pandas as pd
import tensorflow as tf
import datetime
from sklearn.preprocessing import LabelEncoder


def get_weekday(x):
    date = datetime.datetime.strptime(x, '%Y-%m-%d')
    return date.weekday()


def get_time_range(x):
    def get_time(time):
        return datetime.datetime.strptime(time, '%H:%M:%S')

    t = get_time(x)
    if t > get_time('23:00:00'):
        return 23
    else:
        for i in range(0, 24):
            st1 = '0' + str(i) if (i < 10) else str(i)
            st2 = '0' + str(i + 1) if (i + 1 < 10) else str(i + 1)
            if get_time(st1 + ':00' + ':00') < t <= get_time(st2 + ':00' + ':00'):
                return i
    return -1


test_count = 5000

# 读取数据
datas = pd.read_csv('data/BreadBasket_DMS.csv')

# print(datas.head(5))
# print '-----------------------------'
# print('item')
# print(datas['Item'].unique())
# print(datas['Item'].value_counts())
# print '-----------------------------'

features = ['Date', 'Time', 'Item']
all_train_data = datas[features]
print(all_train_data.head(5))

# print '-----------------------------'
all_train_data['Date'] = all_train_data['Date'].map(lambda x: get_weekday(x))
all_train_data['Time'] = all_train_data['Time'].map(lambda x: get_time_range(x))
le_embarked = LabelEncoder()
le_embarked.fit(all_train_data['Item'])
all_train_data['Item'] = le_embarked.transform(all_train_data['Item'])

total = all_train_data.shape[0]
train_data = all_train_data[:total - test_count]
test_data = all_train_data[total - test_count:]
y = datas['Transaction']
train_y = y[:total - test_count]
test_y = y[total - test_count:]
print(train_y.head())