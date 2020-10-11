# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:15:35 2020

@author: yangrun
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing



rain_fall = pd.read_csv('D:\\水电比赛\\遥测站降雨数据.csv',index_col = 0,parse_dates=True)
rain_fall = rain_fall.drop(pd.DatetimeIndex(['2013-01-01 00:00:00', '2013-01-01 01:00:00','2018-01-01 00:00:00', '2018-01-01 01:00:00']),axis =0)
temp1 = rain_fall['2013-01-01 02:00:00':'2017-12-31 23:00:00'].resample('3H', origin='start').sum()
temp2 = rain_fall['2018-01-01 02:00:00':'2018-01-31 23:00:00'].resample('3H', origin='start').sum()
temp3 = rain_fall['2018-07-01 02:00:00':'2018-07-31 23:00:00'].resample('3H', origin='start').sum()
temp4 = rain_fall['2018-10-01 02:00:00':'2018-10-31 23:00:00'].resample('3H', origin='start').sum()

rain_fall = pd.concat([temp1,temp2,temp3,temp4],axis = 0)
rain_fall.sort_index(inplace = True)
 
##########################
#########################
rain_forcast = pd.read_csv('D:\\水电比赛\\降雨预报数据.csv',index_col = 0,parse_dates=True)
#这里补0是因为reasmple里面的一个小bug，如果不多补一行，resample会把最后一天的丢弃。所以多补一行空值.
rain_forcast.loc[pd.Timestamp('2018-02-08'),:] = float(0) 
rain_forcast.loc[pd.Timestamp('2018-08-08'),:] = float(0) 
rain_forcast.loc[pd.Timestamp('2018-11-08'),:] = float(0) 
rain_forcast.sort_index(inplace = True)
temp1 = rain_forcast['2013-01-01 ':'2018-1-1 '].resample('3H', origin=pd.Timestamp('2013-01-01 02:00:00')).ffill().dropna()
temp2 = rain_forcast['2018-01-01' : '2018-02-8 '].resample('3H', origin=pd.Timestamp('2018-01-01 02:00:00')).ffill().dropna()
temp3 = rain_forcast['2018-07-01 ':'2018-08-8 '].resample('3H', origin=pd.Timestamp('2018-07-01 02:00:00')).ffill().dropna()
temp4 = rain_forcast['2018-10-01 ':'2018-11-8 '].resample('3H', origin=pd.Timestamp('2018-10-01 02:00:00')).ffill().dropna()

rain_forcast = pd.concat([temp1,temp2,temp3,temp4],axis = 0)
rain_forcast.sort_index(inplace = True)

#######################
#######################

df_environmnet = pd.read_csv('D:\\水电比赛\\环境表.csv',index_col = 0,parse_dates=True)
####环境表中存在一些#VALUE！将其筛选出来进行
df_environmnet = df_environmnet[~((df_environmnet['w']=='#VALUE!') & (df_environmnet['T']=='#VALUE!'))]
df_environmnet[['w','T']] = df_environmnet[['w','T']].astype('float')

df_environmnet.sort_index(inplace = True)
df_environmnet.dropna(inplace = True)

df_environmnet.loc[pd.Timestamp('2018-02-01'),:] = float(0) 
df_environmnet.loc[pd.Timestamp('2018-08-01'),:] = float(0) 
df_environmnet.loc[pd.Timestamp('2018-11-01'),:] = float(0) 
####这里一定要sort_index
df_environmnet.sort_index(inplace = True)
df_environmnet.dropna(inplace = True)

temp1 = df_environmnet ['2013-01-01 ':'2018-1-1 '].resample('3H', origin=pd.Timestamp('2013-01-01 02:00:00')).ffill().dropna()
temp2 = df_environmnet ['2018-01-01' : '2018-02-1 '].resample('3H', origin=pd.Timestamp('2018-01-01 02:00:00')).ffill().dropna()
temp3 = df_environmnet ['2018-07-01 ':'2018-08-1 '].resample('3H', origin=pd.Timestamp('2018-07-01 02:00:00')).ffill().dropna()
temp4 = df_environmnet ['2018-10-01 ':'2018-11-1 '].resample('3H', origin=pd.Timestamp('2018-10-01 02:00:00')).ffill().dropna()
df_environmnet = pd.concat([temp1,temp2,temp3,temp4],axis = 0)
df_environmnet.sort_index(inplace = True)
###################################################################读取流量数据#######################################
#其中‘Qi’为目标入库流量，其中有333条缺失值2014/11/20 11:00 -2014/12/31 23:00 这里采用2016年对应数据进行补全
df_inflow = pd.read_csv('D:\\水电比赛\\入库流量数据.csv',index_col=0 , parse_dates = True)
temp_value = df_inflow.loc['2016/11/20 11:00':'2016/12/31 23:00',:].values
temp_df = pd.DataFrame(temp,index = pd.date_range('2014/11/20 11:00','2014/12/31 23:00',freq = '3H'),columns = df_inflow.columns)
df_inflow = pd.concat([df_inflow,temp_df],axis = 0)
df_inflow.sort_index(inplace = True)
###################################################################输出所有基本数据####################################
df = pd.concat([df_inflow,rain_fall,rain_forcast,df_environmnet],axis = 1)
df.to_csv('D:\\水电比赛\\new_begin\\df.csv')
