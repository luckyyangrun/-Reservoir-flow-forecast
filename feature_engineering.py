# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:16:49 2020

@author: yangrun
"""


train_df =pd.read_csv('D:\\水电比赛\\new_begin\\df.csv',index_col = 0,parse_dates=True)

###################################################################将wd离散###########################################
train_df =train_df.join(pd.get_dummies(train_df.wd,prefix = 'forward')) 
############################################################把降雨预测滑动到正确位置####################################
for item,num in zip(['D1', 'D2', 'D3', 'D4', 'D5'],range(1,6)):
        column = '{}_ago_forcast'.format(item)
        train_df[column] = train_df[item].shift(num,'1D')
        
train_df.drop(['D1', 'D2', 'D3', 'D4', 'D5','wd'],axis = 1 , inplace = True)


################################################################# 构造时间特征#########################################
def time_period_features(df):
    # in hour level
    df['hour'] = df.index.hour
    # in day level
    df['days_in_month'] = df.index.days_in_month
    df['dayofyear'] = df.index.dayofyear
    # week levels
    df['weekday'] = df.index.weekday
    df['week'] = df.index.week
    df['weekofyear'] = df.index.weekofyear
    # month level
    df['month'] = df.index.month
    df['is_month_start'] = df.index.is_month_start
    df['is_month_start'] = df['is_month_start'].astype(int)
    df['is_month_end'] = df.index.is_month_end
    df['is_month_end'] = df['is_month_end'].astype(int)
    # quarter level
    df['quarter'] = df.index.quarter
    # year level
    df['year'] = df.index.year
    #白天黑夜
    df['night'] = (df.index.hour > 20) | (df.index.hour < 8)
    df['night'] = df['night'].astype(int)
    #日期分钟
    #df['date']= df.index.date
    df['minutes']= df.index.time
    return df


train_df = time_period_features(train_df)
pd.value_counts(train_df['hour'])
train_df =train_df.join(pd.get_dummies(train_df['hour'],prefix = 'hour'))
train_df.drop(['hour'],axis = 1 , inplace = True)

###############################################对流量进行shift，注意shift自带的date-offset里面的m与w并不满足要求(其实应该有更睿智的写法)#######
for num in [1,2,3]:
    column = 'Qi_shift_{}weeks'.format(num)
    temp = train_df[['Qi']].copy()
    temp.index = temp.index + pd.DateOffset(weeks = num)
    temp = temp.rename(columns = {"Qi": column})
    temp = temp[~temp.index.duplicated(keep='first')]
    train_df = pd.concat([train_df,temp],axis = 1)

temp = train_df[['Qi']].copy()
temp.index = temp.index + pd.DateOffset(weeks = 1) + pd.DateOffset(days = 1)
temp = temp.rename(columns = {"Qi": "Qi_shift_8days"})
temp = temp[~temp.index.duplicated(keep='first')]
train_df = pd.concat([train_df,temp],axis = 1)


temp = train_df[['Qi']].copy()
temp.index = temp.index + pd.DateOffset(months = 1)
temp = temp.rename(columns = {"Qi": "Qi_shift_1month"})
temp = temp[~temp.index.duplicated(keep='first')]
train_df = pd.concat([train_df,temp],axis = 1)



temp = train_df[['Qi']].copy()
temp.index = temp.index + pd.DateOffset(years = 1)
temp = temp.rename(columns = {"Qi": "Qi_shift_1year"})
temp = temp[~temp.index.duplicated(keep='first')]
train_df = pd.concat([train_df,temp],axis = 1)


temp = train_df[['Qi']].copy()
temp.index = temp.index + pd.DateOffset(years = 1)+ pd.DateOffset(months = 1)
temp = temp.rename(columns = {"Qi": "Qi_shift_13months"})
temp = temp[~temp.index.duplicated(keep='first')]
train_df = pd.concat([train_df,temp],axis = 1)


temp = train_df[['Qi']].copy()
temp.index = temp.index + pd.DateOffset(years = 1)- pd.DateOffset(months = 1)
temp = temp.rename(columns = {"Qi": "Qi_shift_11months"})
temp = temp[~temp.index.duplicated(keep='first')]
train_df = pd.concat([train_df,temp],axis = 1)

temp = train_df[['Qi']].copy()
temp.index = temp.index + pd.DateOffset(years = 1)+ pd.DateOffset(weeks = 1)
temp = temp.rename(columns = {"Qi": "Qi_shift_1y_plus_1w"})
temp = temp[~temp.index.duplicated(keep='first')]
train_df = pd.concat([train_df,temp],axis = 1)


temp = train_df[['Qi']].copy()
temp.index = temp.index + pd.DateOffset(years = 1)- pd.DateOffset(weeks = 1)
temp = temp.rename(columns = {"Qi": "Qi_shift_1y_minus_1w"})
temp = temp[~temp.index.duplicated(keep='first')]
train_df = pd.concat([train_df,temp],axis = 1)

##上面的脑瘫写法会导致很多冗余数据，这一条将其消除
train_df = train_df.loc[:'2018-11-7',:]

##############################################
########################################################对同期降雨进行求和生成sum_rain，之后分别shift##############################
rain_list = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9',
       'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19',
       'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29',
       'R30', 'R31', 'R32', 'R33', 'R34', 'R35', 'R36', 'R37', 'R38', 'R39']
train_df['sum_rain'] = 0
for item in rain_list:
    train_df['sum_rain']+= train_df[item]

rain_list2 = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9',
       'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19',
       'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29',
       'R30', 'R31', 'R32', 'R33', 'R34', 'R35', 'R36', 'R37', 'R38', 'R39','sum_rain']

for num in [1,2,3]:
    for item in rain_list2:
        temp = train_df[[item]].copy()
        temp.index = temp.index+pd.DateOffset(weeks = num)
        temp = temp.rename(columns = {item: "{}_shift_{}_weeks".format(item,num)})
        temp = temp[~temp.index.duplicated(keep='first')]
        train_df = pd.concat([train_df,temp],axis = 1)


for item in rain_list2:
    temp = train_df[[item]].copy()
    temp.index = temp.index+pd.DateOffset(months = 1)
    temp = temp.rename(columns = {item: "{}_shift_1month".format(item)})
    temp = temp[~temp.index.duplicated(keep='first')]
    train_df = pd.concat([train_df,temp],axis = 1)


train_df.drop(rain_list2,inplace = True,axis = 1)  
train_df = train_df.loc[:'2018-11-7',:]
######################################
##########################################################对特征进行简单加减####################################################

train_df.loc[:,'week_rolling_min'] = [train_df.loc[edt - pd.DateOffset(weeks=1):edt, 'Qi'].min() for edt in train_df.index]
train_df.loc[:,'week_rolling_max'] = [train_df.loc[edt - pd.DateOffset(weeks=1):edt, 'Qi'].max() for edt in train_df.index]
train_df.loc[:,'week_rolling_median'] = [train_df.loc[edt - pd.DateOffset(weeks=1):edt, 'Qi'].median() for edt in train_df.index]
train_df.loc[:,'month_rolling_max'] = [train_df.loc[edt - pd.DateOffset(months=1):edt, 'Qi'].max() for edt in train_df.index]
train_df.loc[:,'month_rolling_median'] = [train_df.loc[edt - pd.DateOffset(months=1):edt, 'Qi'].median() for edt in train_df.index]

train_df.loc[:,'sum_rain_rolling_min'] = [train_df.loc[edt - pd.DateOffset(weeks=1):edt, 'sum_rain_shift_1_weeks'].min() for edt in train_df.index]
train_df.loc[:,'sum_rain_rolling_max'] = [train_df.loc[edt - pd.DateOffset(weeks=1):edt, 'sum_rain_shift_1_weeks'].max() for edt in train_df.index]
train_df.loc[:,'sum_rain_rolling_median'] = [train_df.loc[edt - pd.DateOffset(weeks=1):edt, 'sum_rain_shift_1_weeks'].median() for edt in train_df.index]


train_df.loc[:,'1wQi_diff_min'] = train_df.loc[:,'Qi_shift_1weeks'] - train_df.loc[:,'week_rolling_min']
train_df.loc[:,'1wQi_diff_max'] = train_df.loc[:,'Qi_shift_1weeks'] - train_df.loc[:,'week_rolling_max']
train_df.loc[:,'1wQi_diff_median'] = train_df.loc[:,'Qi_shift_1weeks'] - train_df.loc[:,'week_rolling_median']

train_df.loc[:, "Qi_shift_1y_plus_1w_diff_min"] = train_df.loc[:,"Qi_shift_1y_plus_1w"] - train_df.loc[:,'week_rolling_min']
train_df.loc[:, "Qi_shift_1y_plus_1w_diff_max"] = train_df.loc[:,"Qi_shift_1y_plus_1w"] - train_df.loc[:,'week_rolling_max']
train_df.loc[:, "Qi_shift_1y_plus_1w_diff_median"] = train_df.loc[:,"Qi_shift_1y_plus_1w"] - train_df.loc[:,'week_rolling_median']

train_df.loc[:, "Qi_shift_1y_minus_1w_diff_min"] = train_df.loc[:,"Qi_shift_1y_minus_1w"] - train_df.loc[:,'week_rolling_min']
train_df.loc[:, "Qi_shift_1y_minus_1w_diff_max"] = train_df.loc[:,"Qi_shift_1y_minus_1w"] - train_df.loc[:,'week_rolling_max']
train_df.loc[:, "Qi_shift_1y_minus_1w_diff_median"] = train_df.loc["Qi_shift_1y_minus_1w"] - train_df.loc['week_rolling_median']

train_df.loc[:,'is_odd'] = train_df['year']%2

#############################################################################################################################
############################################对风向进行离散化，并与降雨进行交叉，表示何方来雨#####################################
forward_list = ['forward_999001.0', 'forward_999002.0', 'forward_999003.0',
       'forward_999004.0', 'forward_999005.0', 'forward_999006.0',
       'forward_999007.0', 'forward_999008.0', 'forward_999009.0',
       'forward_999010.0', 'forward_999011.0', 'forward_999012.0',
       'forward_999013.0', 'forward_999014.0', 'forward_999015.0',
       'forward_999016.0']
list_sumrain = ['sum_rain_shift_1_weeks']
for p1 in forward_list:
    for p2 in list_sumrain:
        column = '{}_{}'.format(p1,p2)
        train_df[column] = train_df[p1]*train_df[p2]
        
###################################################
##################################################特征名称有lgb不支持字符，这里替换一下。非lgb模型可以忽略（但不建议）##################
train_df.sort_index(inplace = True)
import re
train_df = train_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))   


train_data = train_df[:'2017-12-31'].dropna().copy()
test_data = train_df['2018-1-1':'2018-11-7'].dropna().copy()

train_data.to_csv('D:\\水电比赛\\new_begin\\train_data.csv')
test_data.to_csv('D:\\水电比赛\\new_begin\\test_data.csv')