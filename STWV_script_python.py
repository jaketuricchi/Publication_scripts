# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:25:12 2020

@author: jaketuricchi
"""
# The aim of the present analysis is to consider the effect of initial body weight variability (BWV)
# on longer-term weight outcomes in participants of the NoHoW weight management
# trial. We will quantify BWV across 3 short term periods (6-, 9- and 12 weeks), and
# consider how well these predict weight change at 6-, 12- and 18 months using
# multivariate linear regression

#%% Import packages
import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings 
import seaborn as sns
import sklearn
from datetime import datetime
import calendar

#%% set wd and read in data

os.chdir(r"C:/Users/jaket/Dropbox/PhD/NoHoW Analyses/Weight variability/Study 5- STWV predict weight outcome")

weights = pd.read_csv('C:/Users/jaket/Dropbox/PhD/NoHoW Analyses/Weight variability/Daily_data_with_NAs_250919.csv')
nh_data = pd.read_csv('C:/Users/jaket/Dropbox/PhD/NoHoW Analyses/Weight variability/all_nh_data_220120.csv', encoding = "ISO-8859-1") 


#%% Macbook setup

#os.chdir(r"/Users/jakw/Dropbox/PhD/NoHoW Analyses/Weight variability/Study 5- STWV predict weight outcome")

#weights = pd.read_csv('/Users/jakw/Dropbox/PhD/NoHoW Analyses/Weight variability/Daily_data_with_NAs_250919.csv')
#nh_data = pd.read_csv('/Users/jakw/Dropbox/PhD/NoHoW Analyses/Weight variability/all_nh_data_220120.csv', encoding = "ISO-8859-1") 




#%% sorting eligibility of weight data by defining minimum length and minimum weights

#first, select the columns we need
weights = weights.filter(items=['ID', 'date', 'day_no', 'weight'])

# we won't be working with any data > 18 months (547 days)
weights = weights[weights['day_no'] < 547]

#get n_weights and length
def define_criteria(x):
    x_complete=x.dropna()
    x['end_day']=x_complete['day_no'].max() #get the maximum day no (length)
    x['n_weights']=len(x_complete) # get number of complete data points
    return(x)

weights = weights.groupby('ID').apply(define_criteria)

#filter by duration and n_weights: we create 3 dfs: 6, 9 and 12 week WV
weights6 = weights[(weights['day_no'] < 43)].groupby('ID').apply(define_criteria)
weights6 = weights6[(weights6['end_day']> 34) & (weights6['n_weights']>9)]
weights6['WV_duration']=6

weights9 = weights[(weights['day_no'] < 64)].groupby('ID').apply(define_criteria)
weights9 = weights9[(weights9['end_day']> 56) & (weights9['n_weights']>15)]
weights9['WV_duration']=9

weights12 = weights[(weights['day_no'] < 85)].groupby('ID').apply(define_criteria)
weights12 = weights12[(weights12['end_day']> 76) & (weights12['n_weights']>19)]
weights12['WV_duration']=12



#%% Calculating weight variability using linear, non-linear (loess) and #
# successive methods
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from numpy import mean, absolute 

def calculate_WV(x):
    x_complete=x.dropna()
    
    X= x_complete['day_no'].values.reshape(-1, 1) #get arrays in correct format
    y= x_complete['weight'].values.reshape(-1, 1)
    
    #RMSE
    lm = LinearRegression()  
    lm=lm.fit(X, y) # fit lm
    ypred=lm.predict(X) #predict
    relative_residuals=((y-ypred)/y)*100 #convert residuals to % residuals
    rmse=np.sqrt((relative_residuals **2).mean()) #get rmse
    
    #NLMD
    lowess = sm.nonparametric.lowess
    loess= lowess(x_complete['weight'], x_complete['day_no'])
    loess=pd.DataFrame([(loess[:,1])]).T
    loess_residual=absolute((y-loess)/y)*100
    nlmd = pd.DataFrame([loess_residual.mean()])
    
    #MASWV
    x_complete['weight_change']=((x_complete['weight'] - x_complete['weight'].shift(1))/x_complete['weight'])*100
    maswv=absolute(x_complete['weight_change'].dropna()).mean()
    
    x['rmse']=rmse
    x['nlmd']=nlmd.loc[0,0]
    x['maswv']=maswv
    
    wv=x.filter(items=['rmse', 'nlmd', 'maswv']).iloc[:1]
    
    return(wv)

#run WV fn to get 3 short term WV dfs
wv6 = weights6.groupby('ID').apply(calculate_WV).reset_index().dropna()
wv9 = weights9.groupby('ID').apply(calculate_WV).reset_index().dropna()
wv12 = weights12.groupby('ID').apply(calculate_WV).reset_index().dropna()

#add a categorical variable defining the WV period for after we concat
wv6['WV_duration']=6
wv9['WV_duration']=9
wv12['WV_duration']=12




#%% removing outliers (+/-5sd, as per Benson&Lowe 2020)

def remove_outliers(x):
    std_rmse=np.array(x['rmse']).std()
    sdx5_rmse=5*std_rmse
    sdd5_rmse=std_rmse/5
    
    std_nlmd=np.array(x['nlmd']).std()
    sdx5_nlmd=5*std_nlmd
    sdd5_nlmd=std_nlmd/5
    
    std_maswv=np.array(x['maswv']).std()
    sdx5_maswv=5*std_maswv
    sdd5_maswv=std_maswv/5
    
    x=x[(x['rmse']<sdx5_rmse) & (x['rmse']>sdd5_rmse)&
        (x['nlmd']<sdx5_nlmd) & (x['nlmd']>sdd5_nlmd)&
        (x['maswv']<sdx5_maswv) & (x['maswv']>sdd5_maswv)]
    return(x)

wv6=remove_outliers(wv6)
wv9=remove_outliers(wv9)
wv12=remove_outliers(wv12)




#%% join WV dfs

wv_all=pd.concat([wv6, wv9, wv12]).reset_index().drop('level_1', axis=1)

#we only want ppts with data in all 3 groups
wv_elig=wv_all.groupby('ID').count()
wv_elig=wv_elig[wv_elig['WV_duration'] >2]

#subsetting wv_all by those eligible
wv_all = wv_all[wv_all['ID'].isin(wv_elig.index)]
weights=weights[weights['ID'].isin(wv_elig.index)]




#%% define day closest to 6, 12 and 18m

#this function finds the closest day to 6, 12 and 18 months with available data
# and returns (a) the weight at that date and (b) the distance from the closest
#  available date and the intended date.
def get_closest_ends(x):
    x_complete=x.dropna()

    weight_6m_index=min(x_complete['day_no'], key=lambda x:abs(x-182))
    weight_6m=x[x['day_no']==weight_6m_index]
    weight_6m['correct_day']=182
    weight_6m['day_diff']= (weight_6m['day_no']- weight_6m['correct_day'])
    
    weight_12m_index=min(x_complete['day_no'], key=lambda x:abs(x-365))
    weight_12m=x[x['day_no']==weight_12m_index]
    weight_12m['correct_day']=365
    weight_12m['day_diff']= (weight_12m['day_no']- weight_12m['correct_day'])
    
    weight_18m_index=min(x_complete['day_no'], key=lambda x:abs(x-547))
    weight_18m=x[x['day_no']==weight_18m_index]
    weight_18m['correct_day']=547
    weight_18m['day_diff']= (weight_18m['day_no']- weight_18m['correct_day'])
    
    weight_ends=pd.concat([weight_6m, weight_12m, weight_18m], axis=0)
    weight_ends=weight_ends.filter(items=['ID', 'weight', 'day_diff'])
    weight_ends['cid']=['cid6', 'cid12', 'cid18']   
    
    end_wide=weight_ends.melt(id_vars='ID').pivot(columns='ID')              
    end_wide=weight_ends.pivot(index='ID',columns='cid', values=['weight', 'day_diff'])
    end_wide.columns = [' '.join(col).strip() for col in end_wide.columns.values]
    end_wide.columns = ['weight.cid12', 'weight.cid18', 'weight.cid6',
                        'day_diff.cid12', 'day_diff.cid18', 'day_diff.cid6']
    end_wide=end_wide.reset_index().drop('ID', axis=1)
    
    return(end_wide)

    
ends=weights.groupby('ID').apply(get_closest_ends).reset_index().drop('level_1', axis=1)

#now we filter participants who have too great
ends = ends[(ends['day_diff.cid6'] > -15) & (ends['day_diff.cid6'] < 15) &
            (ends['day_diff.cid12'] > -15) & (ends['day_diff.cid12'] < 15) &
            (ends['day_diff.cid18'] > -15) & (ends['day_diff.cid18']  < 15)]




#%% In order to create temporal seperation between predictor (WV) and outcome (longer weight change)
# we must calculate weight change beginning after the end of the WV period.

#first, a function will get the weight at the ends of the WV period and calculate the
# change during the WV period (covariate)

def get_weight_change_start_points (x):
    x_complete=x.dropna()
    
    start=x_complete.head(1)           # get start and end weights
    end=x_complete.tail(1)
    start_weight=start['weight'].iloc[0]
    end_weight=end['weight'].iloc[0]
    
    lastweight=pd.DataFrame(end['weight'])           #change will be calculated from the last weight
    lastweight.columns=['last_weight']
    lastweight['WV_duration']=x['WV_duration'].iloc[0]
    lastweight['change']= ((end_weight-start_weight)/start_weight)*100         #get change during the period
    
    return(lastweight)

last_weights6 = weights6.groupby('ID').apply(get_weight_change_start_points).reset_index().drop('level_1', axis=1)
last_weights9 = weights9.groupby('ID').apply(get_weight_change_start_points).reset_index().drop('level_1', axis=1)
last_weights12 = weights12.groupby('ID').apply(get_weight_change_start_points).reset_index().drop('level_1', axis=1)

last_weights=pd.concat([last_weights6, last_weights9, last_weights12])




#%% begin to merge components

analysis_df = pd.merge(wv_all, ends, on='ID')
analysis_df2 = pd.merge(analysis_df, last_weights, on=['ID', 'WV_duration'])
    
#calculate relative weight change between the ends

def weight_change_fn (x):

    y6=x[x['WV_duration']==6]
    y6['rWC_6w6m']=((y6['weight.cid6']-y6['last_weight'])/y6['last_weight'])*100
    y6['rWC_6w12m']=((y6['weight.cid12']-y6['last_weight'])/y6['last_weight'])*100
    y6['rWC_6w18m']=((y6['weight.cid18']-y6['last_weight'])/y6['last_weight'])*100
    
    y12=x[x['WV_duration']==9]
    y12['rWC_9w6m']=((y12['weight.cid6']-y12['last_weight'])/y12['last_weight'])*100
    y12['rWC_9w12m']=((y12['weight.cid12']-y12['last_weight'])/y12['last_weight'])*100
    y12['rWC_9w18m']=((y12['weight.cid18']-y12['last_weight'])/y12['last_weight'])*100
    
    y18=x[x['WV_duration']==12]
    y18['rWC_12w6m']=((y18['weight.cid6']-y18['last_weight'])/y18['last_weight'])*100
    y18['rWC_12w12m']=((y18['weight.cid12']-y18['last_weight'])/y18['last_weight'])*100
    y18['rWC_12w18m']=((y18['weight.cid18']-y18['last_weight'])/y18['last_weight'])*100
    
    y=pd.concat([y6, y12, y18])
    y=y.fillna(y.mean()).drop('ID', axis=1)
    
    return(y)

analysis_df3 = analysis_df2.groupby('ID').apply(weight_change_fn).reset_index().drop('level_1', axis=1)
         
                                                                             
                                                                                      
#%% get additional variables for model covariates

# age, gender, trial arm, weight suppression
covariates=nh_data.filter(items=['participant_study_id', 'study_arm_id',
                                 'elig_age', 'elig_gender',
                                 'ecid1_highest_weight_12', 'ecid1_lowest_weight_12'])
covariates['rWL']=covariates['ecid1_highest_weight_12']-covariates['ecid1_lowest_weight_12']
covariates=covariates.drop(['ecid1_highest_weight_12', 'ecid1_lowest_weight_12'], axis=1)
covariates.columns=['ID', 'arm', 'age', 'gender', 'rWL']

#determine scale use for each of the periods. e.g. non-missing data points used to calculate WV
n_weights6=weights6.groupby(['ID', 'WV_duration']).n_weights.agg('mean').reset_index()
n_weights9=weights9.groupby(['ID', 'WV_duration']).n_weights.agg('mean').reset_index()
n_weights12=weights12.groupby(['ID', 'WV_duration']).n_weights.agg('mean').reset_index()
n_weights=pd.concat([n_weights6, n_weights9, n_weights12])

#merge this into covariates (will create 3 rows per ppt for each WV_duration)
covariates=pd.merge(covariates, n_weights, on='ID')

#merge additional covs back into analysis df
analysis_df4=pd.merge(analysis_df3, covariates, on=['ID', 'WV_duration']).dropna().reset_index().drop('level_0', axis=1)




#%% Scale WV measurements
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#we'll scale on the 1 scale, so melt, scale, cast
wv_to_scale=analysis_df4.filter(items=['ID', 'WV_duration', 'nlmd', 'rmse', 'maswv']).melt(id_vars=['ID', 'WV_duration'])

#scale
wv_to_scale['value'] = scaler.fit_transform(wv_to_scale['value'].values.reshape(-1,1))

#cast
wv_to_scale_wide=wv_to_scale.pivot_table(index=['ID','WV_duration'], 
                                   columns='variable', values='value').reset_index()
wv_to_scale_wide.columns=['ID', 'WV_duration', 'maswv_s', 'nlmd_s', 'rmse_s']


#merge the new scaled WV measures into WV df and delete the unscaled vars
analysis_df5=pd.merge(analysis_df4, wv_to_scale_wide, on=['ID','WV_duration']).drop(['nlmd','rmse','maswv'], axis=1)




#%% visualisation of main effects

#a bit of preprocessing the structure for this plot
plot1_wc=analysis_df5.filter(items=['ID', 'WV_duration','rWC_6w6m', 'rWC_6w12m', 'rWC_6w18m', 'rWC_9w6m', 
                              'rWC_9w12m','rWC_9w18m', 'rWC_12w6m', 'rWC_12w12m', 
                              'rWC_12w18m'])
plot1_wc_long=pd.melt(plot1_wc, id_vars=['ID','WV_duration'])    
plot1_wc_long.columns=['ID','WV_duration','WC_comparison','WC']

plot1_wv=analysis_df5.filter(items=['ID', 'WV_duration', 'maswv_s', 'rmse_s','nlmd_s'])
plot1_wv_long=pd.melt(plot1_wv, id_vars=['ID','WV_duration'])
plot1_wv_long.columns=['ID','WV_duration','WV_method','WV']

plot1_df=pd.merge(plot1_wc_long, plot1_wv_long, on=['ID','WV_duration'])

plot1_df['wv_duration'] = np.where(
        plot1_df['WC_comparison']. str.contains('6w'), 6,
        np.where(plot1_df['WC_comparison'].str.contains('9w'),9,12))
plot1_df['rWC_duration'] = np.where(
        plot1_df['WC_comparison']. str.contains('6m'), 6,
        np.where(plot1_df['WC_comparison'].str.contains('12m'),12,18))

plot1_df['wv_match']=np.where(
        plot1_df['WV_duration']==plot1_df['wv_duration'], 1, 0)

plot1_df=plot1_df[plot1_df['wv_match']==1]

# plot
#g = sns.PairGrid(plot1_df, vars=["WV", "WC"], hue="WV_method")
g = sns.FacetGrid(plot1_df, col="wv_duration",  row="rWC_duration", hue="WV_method")
g = g.map(sns.regplot, "WV", "WC")
g.set(ylim=(-30,30), xlim=(0,1))
g.set_axis_labels(xlabel='Standardized weight variability', ylabel='Percent weight change',fontsize=15 )



#%% additional visualisation for manuscript/supplementary figures

#Plot2:  association between baseline weight and weight change at 18m
plot2_df=nh_data.filter(items=['participant_study_id', 'ecid1_weight_recorded',
                               'ecid2_weight_recorded', 'ecid3_weight_recorded',
                               'ecid4_weight_recorded'])

plot2_df.columns=['ID','weight0','weight6', 'weight12', 'weight18']
plot2_df=plot2_df[plot2_df['ID'].isin(analysis_df5['ID'])]


plot2_df['rWC6']=((plot2_df['weight6']-plot2_df['weight0'])/plot2_df['weight0'])*100
plot2_df['rWC12']=((plot2_df['weight12']-plot2_df['weight0'])/plot2_df['weight0'])*100
plot2_df['rWC18']=((plot2_df['weight18']-plot2_df['weight0'])/plot2_df['weight0'])*100

plot2_df=plot2_df.drop(['weight6','weight12','weight18'], axis=1)
plot2_df_long=pd.melt(plot2_df, id_vars=['ID', 'weight0'])

plot2_df_long.variable = plot2_df_long.variable.astype('category') 
plot2_df_long.variable = plot2_df_long['variable'].replace(
        {'rWC6': 'Perc_WC_6m', 'rWC12': 'Perc_WC_12m',
         'rWC18': 'Perc_WC_18m'})

plot2=sns.lmplot(x="weight0", y="value", hue="variable", scatter_kws={"s": 8},
           ci=None,data=plot2_df_long)
plot2.set(xlabel='Baseline weight (kg)', ylabel='Percentage weight change')


#%% Plot 3: association between weight suppression (rWL) and weight change 

plot3_df=covariates.filter(items=['ID', 'rWL'])
plot3_df=pd.merge(plot3_df, plot2_df, on='ID').drop('weight0', axis=1)
plot3_df=plot3_df.groupby('ID').first().reset_index()
plot3_df_long=pd.melt(plot3_df, id_vars=['ID','rWL'])

plot3=sns.lmplot(x="rWL", y="value", hue="variable", scatter_kws={"s": 8},
           ci=None,data=plot3_df_long)
plot3.set(xlabel='12m Weight Suppression (kg)', ylabel='Percentage weight change')



#%% Plot 4: change in scale use over the period
weights=weights[weights['ID'].isin(analysis_df5['ID'])]

def ceiling(a, b):
    return -(-a // b)

weights['week_no']=ceiling(weights['day_no'], 7)

plot4_df=weights.groupby(['ID', 'week_no'])['weight'].count().reset_index()
plot4_df=plot4_df.groupby('week_no')['weight'].mean().reset_index()

plot4=sns.lineplot(x='week_no', y='weight', data=plot4_df, err_style='bars')
plot4.set(xlabel='Week of trial', ylabel='Scale use (time per week)')
plot4.set(ylim=(2,5))



#%%plot 5a: scale use per dow
weights['date'] = pd.to_datetime(weights['date'])

weights['weekday_no']=weights['date'].dt.dayofweek.astype('category')
weights['weekday']=weights['weekday_no'].replace(
        {0:"Monday", 1:"Tuesday",2:"Wednesday",3:"Thursday", 
         4:"Friday",5:"Saturday",6:"Sunday"})
    
plot5_df = weights.groupby(['ID','weekday']).count().reset_index()
plot5_df['dow_completeness']= (plot5_df['weight']/plot5_df['day_no'])*100
plot5_df=plot5_df.groupby('weekday')['dow_completeness'].mean().reset_index()

plot5=sns.barplot(x='dow_completeness', y='weekday', data=plot5_df,
                  order=["Monday", "Tuesday", 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                         'Sunday'], ci="sd")
plot5.set(xlabel='Weekday', ylabel='Fraction of complete data (%)')

#%% Plot 5b

weights['month']= weights['date'].dt.strftime('%B')

    
plot6_df = weights.groupby(['ID','month']).count().reset_index()
plot6_df['month_completeness']= (plot6_df['weight']/plot6_df['day_no'])*100
plot6_df=plot6_df.groupby('month')['month_completeness'].mean().reset_index()

plot6=sns.barplot(x='month_completeness', y='month', data=plot6_df,
                  order=["January", "February", 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October',
                         'November', 'December'], ci="sd")
plot6.set(xlabel='Month', ylabel='Fraction of complete data (%)')

#%% Main analysis - crude regressions
import statsmodels.api as sm

def get_model1_results (X, Y, duration):
    predictor=pd.DataFrame([X.columns])
    outcome=pd.DataFrame([Y.name])
    Y=list(Y)
    X=pd.concat([X, duration], axis=1).reset_index(drop=True)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    r2=pd.DataFrame([[model.rsquared]]).iloc[0]
    res=pd.concat([model.params,model.bse, model.pvalues], axis=1).iloc[[1]].reset_index(drop=True)
    res2=pd.concat([predictor, res, r2, outcome], axis=1)
    res2.columns=['predictor', 'coefficient', 'std_err', 'pvalue', 'r2', 'outcome']
    return(res2)


    
#results for WV=6
#nlmd
lm_crude_6=analysis_df5[analysis_df5['WV_duration']==6]
mod1_wv6_results=get_model1_results(lm_crude_6[['nlmd_s']], lm_crude_6['rWC_6w6m'], lm_crude_6[['day_diff.cid6']])
mod1_wv6_results=pd.concat([mod1_wv6_results, get_model1_results(lm_crude_6[['nlmd_s']], lm_crude_6['rWC_6w12m'], lm_crude_6[['day_diff.cid12']])], axis=0)
mod1_wv6_results=pd.concat([mod1_wv6_results, get_model1_results(lm_crude_6[['nlmd_s']], lm_crude_6['rWC_6w18m'], lm_crude_6[['day_diff.cid18']])], axis=0)
#rmse
mod1_wv6_results=pd.concat([mod1_wv6_results, get_model1_results(lm_crude_6[['rmse_s']], lm_crude_6['rWC_6w6m'], lm_crude_6[['day_diff.cid6']])], axis=0)
mod1_wv6_results=pd.concat([mod1_wv6_results, get_model1_results(lm_crude_6[['rmse_s']], lm_crude_6['rWC_6w12m'], lm_crude_6[['day_diff.cid12']])], axis=0)
mod1_wv6_results=pd.concat([mod1_wv6_results, get_model1_results(lm_crude_6[['rmse_s']], lm_crude_6['rWC_6w18m'], lm_crude_6[['day_diff.cid18']])], axis=0)
mod1_wv6_results['WV_duration']=6

#results for WV=9
#nlmd
lm_crude_9=analysis_df5[analysis_df5['WV_duration']==9]
mod1_wv9_results=get_model1_results(lm_crude_9[['nlmd_s']], lm_crude_9['rWC_9w6m'], lm_crude_9[['day_diff.cid6']])
mod1_wv9_results=pd.concat([mod1_wv9_results, get_model1_results(lm_crude_9[['nlmd_s']], lm_crude_9['rWC_9w12m'], lm_crude_9[['day_diff.cid12']])], axis=0)
mod1_wv9_results=pd.concat([mod1_wv9_results, get_model1_results(lm_crude_9[['nlmd_s']], lm_crude_9['rWC_9w18m'], lm_crude_9[['day_diff.cid18']])], axis=0)
#rmse
mod1_wv9_results=pd.concat([mod1_wv9_results, get_model1_results(lm_crude_9[['rmse_s']], lm_crude_9['rWC_9w6m'], lm_crude_9[['day_diff.cid6']])], axis=0)
mod1_wv9_results=pd.concat([mod1_wv9_results, get_model1_results(lm_crude_9[['rmse_s']], lm_crude_9['rWC_9w12m'], lm_crude_9[['day_diff.cid12']])], axis=0)
mod1_wv9_results=pd.concat([mod1_wv9_results, get_model1_results(lm_crude_9[['rmse_s']], lm_crude_9['rWC_9w18m'], lm_crude_9[['day_diff.cid18']])], axis=0)
mod1_wv9_results['WV_duration']=9

#results for WV=12
#nlmd
lm_crude_12=analysis_df5[analysis_df5['WV_duration']==12]
mod1_wv12_results=get_model1_results(lm_crude_12[['nlmd_s']], lm_crude_12['rWC_12w6m'], lm_crude_12[['day_diff.cid6']])
mod1_wv12_results=pd.concat([mod1_wv12_results, get_model1_results(lm_crude_12[['nlmd_s']], lm_crude_12['rWC_12w12m'], lm_crude_12[['day_diff.cid12']])], axis=0)
mod1_wv12_results=pd.concat([mod1_wv12_results, get_model1_results(lm_crude_12[['nlmd_s']], lm_crude_12['rWC_12w18m'], lm_crude_12[['day_diff.cid18']])], axis=0)
#rmse
mod1_wv12_results=pd.concat([mod1_wv12_results, get_model1_results(lm_crude_12[['rmse_s']], lm_crude_12['rWC_12w6m'], lm_crude_12[['day_diff.cid6']])], axis=0)
mod1_wv12_results=pd.concat([mod1_wv12_results, get_model1_results(lm_crude_12[['rmse_s']], lm_crude_12['rWC_12w12m'], lm_crude_12[['day_diff.cid12']])], axis=0)
mod1_wv12_results=pd.concat([mod1_wv12_results, get_model1_results(lm_crude_12[['rmse_s']], lm_crude_12['rWC_12w18m'], lm_crude_12[['day_diff.cid18']])], axis=0)
mod1_wv12_results['WV_duration']=12

mod1_results_all=pd.concat([mod1_wv6_results, mod1_wv9_results, mod1_wv12_results], axis=0)

#%% Main analysis - multivariate regressions
covariates=['rWL', 'arm', 'age', 'gender', 'n_weights', 'last_weight', 'change']

analysis_df5['gender']=np.where(analysis_df5['gender']=='Male',1,0)
analysis_df5['gender']=analysis_df5['gender'].astype('category')

def get_model2_results (df, X, Y, duration):
    covariates_df=df.filter(items=covariates).reset_index(drop=True)
    predictor=pd.DataFrame([X.columns])
    outcome=pd.DataFrame([Y.name])
    Y=list(Y)
    X=pd.concat([X, duration, covariates_df], axis=1).reset_index(drop=True)
    X = sm.add_constant(X) 
    model = sm.OLS(Y, X.astype(float)).fit()
    r2=pd.DataFrame([[model.rsquared]]).iloc[0]
    res=pd.concat([model.params,model.bse, model.pvalues], axis=1).iloc[[1]].reset_index(drop=True)
    res2=pd.concat([predictor, res, r2, outcome], axis=1)
    res2.columns=['predictor', 'coefficient', 'std_err', 'pvalue', 'r2', 'outcome']
    return(res2)

#nlmd
lm_crude_6=analysis_df5[analysis_df5['WV_duration']==6].reset_index(drop=True)
mod2_wv6_results=get_model2_results(lm_crude_6, lm_crude_6[['nlmd_s']], lm_crude_6['rWC_6w6m'], lm_crude_6[['day_diff.cid6']])
mod2_wv6_results=pd.concat([mod1_wv6_results, get_model2_results(lm_crude_6, lm_crude_6[['nlmd_s']], lm_crude_6['rWC_6w12m'], lm_crude_6[['day_diff.cid12']])], axis=0)
mod2_wv6_results=pd.concat([mod1_wv6_results, get_model2_results(lm_crude_6, lm_crude_6[['nlmd_s']], lm_crude_6['rWC_6w18m'], lm_crude_6[['day_diff.cid18']])], axis=0)
#rmse
mod2_wv6_results=pd.concat([mod2_wv6_results, get_model2_results(lm_crude_6, lm_crude_6[['rmse_s']], lm_crude_6['rWC_6w6m'], lm_crude_6[['day_diff.cid6']])], axis=0)
mod2_wv6_results=pd.concat([mod2_wv6_results, get_model2_results(lm_crude_6, lm_crude_6[['rmse_s']], lm_crude_6['rWC_6w12m'], lm_crude_6[['day_diff.cid12']])], axis=0)
mod2_wv6_results=pd.concat([mod2_wv6_results, get_model2_results(lm_crude_6, lm_crude_6[['rmse_s']], lm_crude_6['rWC_6w18m'], lm_crude_6[['day_diff.cid18']])], axis=0)
mod2_wv6_results['WV_duration']=6
    
#results for WV=9
#nlmd
lm_crude_9=analysis_df5[analysis_df5['WV_duration']==9]
mod2_wv9_results=get_model1_results(lm_crude_9[['nlmd_s']], lm_crude_9['rWC_9w6m'], lm_crude_9[['day_diff.cid6']])
mod2_wv9_results=pd.concat([mod2_wv9_results, get_model1_results(lm_crude_9[['nlmd_s']], lm_crude_9['rWC_9w12m'], lm_crude_9[['day_diff.cid12']])], axis=0)
mod2_wv9_results=pd.concat([mod2_wv9_results, get_model1_results(lm_crude_9[['nlmd_s']], lm_crude_9['rWC_9w18m'], lm_crude_9[['day_diff.cid18']])], axis=0)
#rmse
mod2_wv9_results=pd.concat([mod2_wv9_results, get_model1_results(lm_crude_9[['rmse_s']], lm_crude_9['rWC_9w6m'], lm_crude_9[['day_diff.cid6']])], axis=0)
mod2_wv9_results=pd.concat([mod2_wv9_results, get_model1_results(lm_crude_9[['rmse_s']], lm_crude_9['rWC_9w12m'], lm_crude_9[['day_diff.cid12']])], axis=0)
mod2_wv9_results=pd.concat([mod2_wv9_results, get_model1_results(lm_crude_9[['rmse_s']], lm_crude_9['rWC_9w18m'], lm_crude_9[['day_diff.cid18']])], axis=0)
mod2_wv9_results['WV_duration']=9

#results for WV=12
#nlmd
lm_crude_12=analysis_df5[analysis_df5['WV_duration']==12]
mod2_wv12_results=get_model1_results(lm_crude_12[['nlmd_s']], lm_crude_12['rWC_12w6m'], lm_crude_12[['day_diff.cid6']])
mod2_wv12_results=pd.concat([mod2_wv12_results, get_model1_results(lm_crude_12[['nlmd_s']], lm_crude_12['rWC_12w12m'], lm_crude_12[['day_diff.cid12']])], axis=0)
mod2_wv12_results=pd.concat([mod2_wv12_results, get_model1_results(lm_crude_12[['nlmd_s']], lm_crude_12['rWC_12w18m'], lm_crude_12[['day_diff.cid18']])], axis=0)
#rmse
mod2_wv12_results=pd.concat([mod2_wv12_results, get_model1_results(lm_crude_12[['rmse_s']], lm_crude_12['rWC_12w6m'], lm_crude_12[['day_diff.cid6']])], axis=0)
mod2_wv12_results=pd.concat([mod2_wv12_results, get_model1_results(lm_crude_12[['rmse_s']], lm_crude_12['rWC_12w12m'], lm_crude_12[['day_diff.cid12']])], axis=0)
mod2_wv12_results=pd.concat([mod2_wv12_results, get_model1_results(lm_crude_12[['rmse_s']], lm_crude_12['rWC_12w18m'], lm_crude_12[['day_diff.cid18']])], axis=0)
mod2_wv12_results['WV_duration']=12

mod2_results_all=pd.concat([mod2_wv6_results, mod2_wv9_results, mod2_wv12_results], axis=0)



#%% Plot results - model1

mod1_results_all['rWC_duration'] = np.where(
        mod1_results_all['outcome']. str.contains('6m'), 6,
        np.where(mod1_results_all['outcome'].str.contains('12m'),12,18))
mod1_results_plot=mod1_results_all.filter(items=['rWC_duration', 'rWC_duration',
                                                 'coefficient', 'predictor', 'std_err']).reset_index(drop=True)
mod1_results_all['WV_duration']=mod1_results_all['WV_duration'].astype('category')
mod1_results_all['rWC_duration']=mod1_results_all['rWC_duration'].astype('category')



mod1_plot = sns.FacetGrid(mod1_results_all, col="WV_duration", row="rWC_duration", despine=True, sharex=False)
mod1_plot = mod1_plot.map(sns.pointplot, data= mod1_results_all, x="coeff", y="predictor", join=False,
                          yerr='std_err')
mod1_plot.set(xlabel='Standardized Regession Coefficient', ylabel='WV method')

with sns.axes_style("white"):
    g = sns.FacetGrid(mod1_results_all, col="WV_duration",  row="rWC_duration", margin_titles=True, height=2.5)
g.map(sns.scatterplot, x="coefficient", y="predictor",  data=mod1_results_all)
#next:add R2 labels, then plot mod2


