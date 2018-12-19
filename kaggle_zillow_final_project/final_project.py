import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
#import datetime as dt
from datetime import datetime

# Parameters
#XGB_WEIGHT = 0.6415
#BASELINE_WEIGHT = 0.0056
#OLS_WEIGHT = 0.0828

#XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

#BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

##### READ IN RAW DATA
#print( "\nReading data from disk ...")
#prop = pd.read_csv('properties_2016.csv')
train = pd.read_csv("train_2016_v2.csv")


################
################
##  XGBoost   ##
################
################
XGB1_WEIGHT = 0.8000
##### RE-READ PROPERTIES FILE
##### (I tried keeping a copy, but the program crashed.)

print( "\nRe-reading properties file ...")
properties = pd.read_csv('properties_2016.csv')

##### PROCESS DATA FOR XGBOOST
print( "\nProcessing data for XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out outliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))


##### RUN XGBOOST
print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,   
    'alpha': 0.4, 
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 250
print("num_boost_rounds="+str(num_boost_rounds))

# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost ...")
xgb_pred1 = model.predict(dtest)

##### RUN XGBOOST AGAIN

print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

num_boost_rounds = 150
print("num_boost_rounds="+str(num_boost_rounds))

print( "\nTraining XGBoost again ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost again ...")
xgb_pred2 = model.predict(dtest)

print( "\nSecond XGBoost predictions:" )
print( pd.DataFrame(xgb_pred2).head() )



##### COMBINE XGBOOST RESULTS
xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2
#xgb_pred = xgb_pred1

print( "\nCombined XGBoost predictions:" )
print( pd.DataFrame(xgb_pred).head() )

prediction=pd.DataFrame(xgb_pred)
prediction.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
print( "\nFirst XGBoost predictions:" )
print( prediction.head(30) )
#print(xgb_pred1.shape)
