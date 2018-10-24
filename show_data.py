# coding=utf-8
import pandas as pd
import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt


def get_weekday(x):
    date = datetime.datetime.strptime(x, '%Y-%m-%d')
    return str(date.weekday())


init_date = datetime.datetime.strptime('2016-10-30', '%Y-%m-%d')


def get_delta_days(x):
    date = datetime.datetime.strptime(x, '%Y-%m-%d')
    delta = date - init_date
    return delta.days / 1000.


def get_festival(x):
    date = datetime.datetime.strptime(x, '%Y-%m-%d')
    if date.strftime('%m-%d').__eq__('01-01'):
        return '元旦节'
    elif date.strftime('%m-%d').__eq__('02-14'):
        return '情人节'
    elif date.strftime('%m-%d').__eq__('04-01'):
        return '愚人节'
    elif date.strftime('%m-%d').__eq__('04-16'):
        return '复活节'
    elif date.strftime('%m-%d').__eq__('05-14'):
        return '母亲节'
    elif date.strftime('%m-%d').__eq__('06-18'):
        return '父亲节'
    elif date.strftime('%m-%d').__eq__('07-04'):
        return '独立日'
    elif date.strftime('%m-%d').__eq__('10-31'):
        return '万圣节'
    elif date.strftime('%m-%d').__eq__('11-01'):
        return '万圣节'
    elif date.strftime('%m-%d').__eq__('12-24'):
        return '圣诞节'
    elif date.strftime('%m-%d').__eq__('12-25'):
        return '圣诞节'
    else:
        return 'null'


def get_time_range(x):
    def get_time(time):
        return datetime.datetime.strptime(time, '%H:%M:%S')

    t = get_time(x)
    if t > get_time('23:00:00'):
        return str(23)
    else:
        for i in range(0, 24):
            st1 = '0' + str(i) if (i < 10) else str(i)
            st2 = '0' + str(i + 1) if (i + 1 < 10) else str(i + 1)
            if get_time(st1 + ':00' + ':00') < t <= get_time(st2 + ':00' + ':00'):
                return str(i)
    return str(-1)


def get_time_quantum(x):
    def get_time(time):
        return datetime.datetime.strptime(time, '%H:%M:%S')

    t = get_time(x)
    # 早晨
    if get_time('06:00:00') < t <= get_time('08:00:00'):
        return "早晨"
    # 上午
    elif get_time('08:00:00') < t <= get_time('11:00:00'):
        return "上午"
    # 中午
    elif get_time('11:00:00') < t <= get_time('13:00:00'):
        return "中午"
    # 下午
    elif get_time('13:00:00') < t <= get_time('17:00:00'):
        return "下午"
    # 傍晚
    elif get_time('17:00:00') < t <= get_time('19:00:00'):
        return "傍晚"
    # 晚上
    elif get_time('19:00:00') < t <= get_time('23:00:00'):
        return "晚上"
    # 深夜
    elif get_time('23:00:00') > t or t <= get_time('03:00:00'):
        return "深夜"
    # 凌晨
    # get_time('03:00:00') < t <= get_time('06:00:00')
    else:
        return "凌晨"


pd.options.mode.chained_assignment = None
# 读取数据
datas = pd.read_csv('data/BreadBasket_DMS.csv')
print("原始数据shape：" + str(datas.shape))
print '--------------数据信息---------------'

# 查看数据情，主要检查有没有缺失数据
datas.info()
# print(datas.head(5))
# print '-----------------------------'
# print('item')
# print(datas['Item'].unique())
# print(datas['Item'].value_counts())
# print '-----------------------------'

# print '--------------去掉结果特征---------------'
# features = ['Date', 'Time', 'Item']
# all_train_data = datas[features]
# print(all_train_data.head())
# ->
#         Date      Time           Item
# 0  2016-10-30  09:58:11          Bread
# 1  2016-10-30  10:05:34   Scandinavian
# 2  2016-10-30  10:05:34   Scandinavian
# 3  2016-10-30  10:07:57  Hot chocolate
# 4  2016-10-30  10:07:57            Jam

# print '--------------Item One-Hot---------------'
# itemDf = pd.get_dummies(datas['Item'], prefix='Item')
# print(itemDf.head())
# new_datas = pd.concat([datas, itemDf], axis=1)
# new_datas.drop('Item', axis=1, inplace=True)
# print(new_datas.shape)


print '--------------绘图---------------'
plt.figure(1)
plt.subplot(221)
plt.plot(datas['Date'], datas['Transaction'], 'ro')
plt.xlabel('Date')
plt.ylabel('Transaction')
plt.legend()

# plt.subplot(222)
# plt.plot(datas['Time'], datas['Transaction'], 'ro')
# plt.xlabel('Time')
# plt.ylabel('Transaction')
# plt.legend()


print '--------------处理数据类型转换---------------'
# 挖掘，周几对交易量的影响
datas['Weekday'] = datas['Date']
datas['Weekday'] = datas['Weekday'].map(lambda x: get_weekday(x))

# 挖掘，各个节日对交易量的影响
datas['Festival'] = datas['Date']
datas['Festival'] = datas['Festival'].map(lambda x: get_festival(x))

# 挖掘，当前所处的时间段对交易量的影响
datas['Time_Quantum'] = datas['Time']
datas['Time_Quantum'] = datas['Time_Quantum'].map(lambda x: get_time_quantum(x))

# 随着日期的增长，交易量在增长，所以日期的大小会影响交易量
datas['Date'] = datas['Date'].map(lambda x: get_delta_days(x))
datas['Time'] = datas['Time'].map(lambda x: get_time_range(x))
datas['Transaction'] = datas['Transaction'].map(lambda x: (float(x) / 1000.))
datas.info()
print(datas.head())

print '--------------One-Hot处理---------------'
ont_hot_data = pd.get_dummies(datas, prefix=['Time', 'Item', 'Weekday', 'Festival', 'Time_Quantum'])


# print '--------------补齐---------------'
# null_count = 144 - ont_hot_data.shape[1] + 1
# null_count = 1
# for i in range(0, null_count):
#     name = 'null_' + str(i)
#     ont_hot_data[name] = datas['Date']
#     ont_hot_data[name] = ont_hot_data[name].map(lambda x: 0.)
# print(ont_hot_data.shape)
# -》 (21293, 145)
# print(ont_hot_data.head())


# print '--------------归一化---------------'
# transaction_min = ont_hot_data['Transaction'].min()
# transaction_max = ont_hot_data['Transaction'].max()
# ont_hot_data['Transaction'] = ont_hot_data['Transaction'].map(
#     lambda x: (x - transaction_min) / (transaction_max - transaction_min))
#
# date_min = ont_hot_data['Date'].min()
# date_max = ont_hot_data['Date'].max()
# ont_hot_data['Date'] = ont_hot_data['Date'].map(
#     lambda x: (x - date_min) / (date_max - date_min))
#
#
# print(ont_hot_data.head())

print '--------------归一化绘图---------------'
plt.subplot(222)
plt.plot(ont_hot_data['Date'], ont_hot_data['Transaction'], 'ro')
plt.xlabel('Date')
plt.ylabel('Transaction')
plt.legend()
plt.show()
# le_embarked = LabelEncoder()
# le_embarked.fit(all_train_data['Item'])
# all_train_data['Item'] = le_embarked.transform(all_train_data['Item'])

print '--------------打乱顺序---------------'
ont_hot_data = ont_hot_data.sample(frac=1, replace=False)
ont_hot_data = ont_hot_data.sample(frac=1, replace=False)
ont_hot_data = ont_hot_data.sample(frac=1, replace=False)
# -》 (21293, 136)
print(ont_hot_data.head())

print '--------------保存新数据---------------'
# 测试数据集大小
test_count = 6000
train_count = ont_hot_data.shape[0] - test_count
# 切割出训练数据集
train_data = ont_hot_data[:train_count]
# 切割出测试数据集
test_data = ont_hot_data[train_count:]
# 分别保存两个数据集
train_data.to_csv('data/train_data.csv', index=False, header=True)
test_data.to_csv('data/test_data.csv', index=False, header=True)
