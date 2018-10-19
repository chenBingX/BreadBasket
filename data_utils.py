# coding=utf-8

import pandas as pd
import tensorflow as tf

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

all_train_data['Date'] = all_train_data['Date'].map(lambda x:)