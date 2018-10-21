# coding=utf-8

import pandas as pd


class BBDATA():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

class DATA():
    def __init__(self,  data):
        self.src = data
        self.label = data['Transaction']
        copy = data.copy()
        copy.drop("Transaction", axis=1, inplace=True)
        self.data = copy

    def batch(self, count):
        batch = self.src.sample(n=count, replace=False)
        return DATA(batch)

def read_datas(path):
    train_data = pd.read_csv(path + 'train_data.csv')
    test_data = pd.read_csv(path + 'test_data.csv')
    return BBDATA(DATA(train_data), DATA(test_data))
