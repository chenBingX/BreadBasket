# coding=utf-8

import datetime
str = '2016-10-30'
date = datetime.datetime.strptime(str, '%Y-%m-%d')
print(date)
print(date.weekday())