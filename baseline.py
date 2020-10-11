# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:17:57 2020

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


train_df = pd.read_csv('D:\\水电比赛\\new_begin\\train_data.csv',index_col=0 , parse_dates = True)
#train_df.drop(['minutes'],axis = 1 , inplace = True)
temp = train_df.loc[(train_df.index.month == 11)|(train_df.index.month == 12)|(train_df.index.month == 1)|(train_df.index.month == 2)|(train_df.index.month == 3), :]

split_dt1 = pd.Timestamp('2017-2-1')
split_dt2 = pd.Timestamp('2017-2-8')
temp1 = temp[:split_dt1]
temp2 = temp[split_dt2:]
train_data = pd.concat([temp1,temp2],axis = 0)
train_data.sort_index(inplace = True)
test_data = temp[split_dt1:split_dt2]




train_data_Y = train_data[['Qi']].copy()
train_data_X = train_data.drop(['Qi'],axis =1)
test_data_Y = test_data[['Qi']].copy()
test_data_X = test_data.drop(['Qi'],axis =1)


lgb_train = lgb.Dataset(train_data_X,train_data_Y)
lgb_eval = lgb.Dataset(test_data_X  ,test_data_Y  ,reference = lgb.train)

params = {
    'boosting':'gbdt',
    'max_depth': 4,
    'num_leaves':8,
    'eta': 0.05,
    'metric': ['rmse'],
    'min_data_in_leaf':20,
    #'lambda_l1':5,
    'bagging_freq': 1}
evals_result = {}
watch_list = [lgb_train,lgb_eval]
valid_name = ['train','eval']
gbm = lgb.train(params,
                lgb_train,
                num_boost_round = 20000,
                valid_sets = watch_list,
                valid_names = valid_name,
               early_stopping_rounds =200,
               evals_result=evals_result)
gbm.save_model('D:\\newbegin_lgb_model.txt')
print('Starting predicting...')
# predict
y_pred = gbm.predict(test_data_X, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', np.sqrt(mean_squared_error(test_data_Y,y_pred )))


y_pred = pd.DataFrame(y_pred,index= test_data_Y.index)
plt.figure(figsize=(20,20))
plt.plot(y_pred ,label = 'pred',color = 'r')
plt.plot(test_data_Y ,label = 'true',color = 'y')
metric = 'rmse'
fig,ax = plt.subplots()
ax.plot(evals_result['train'][metric],label = 'train')
ax.plot(evals_result['eval'][metric],label = 'Test')
ax.legend()
plt.ylabel(f'{metric}')
plt.title(f'LightGBM {metric}')
plt.show()
###############################
################################
from sklearn import linear_model
reg = linear_model.Ridge(alpha = 0.2)
reg.fit(train_data_X,train_data_Y)

y_pred2 = reg.predict(test_data_X)
print('The rmse of prediction is:', np.sqrt(mean_squared_error(test_data_Y,y_pred2 )))
y_pred2 = pd.DataFrame(y_pred2,index= test_data_Y.index)
plt.figure(figsize=(20,20))
plt.plot(y_pred2 ,label = 'pred',color = 'r')
plt.plot(test_data_Y ,label = 'true',color = 'y')
