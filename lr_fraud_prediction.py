import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df = pd.read_csv('traindataset.csv')

#Preprocessing
x_columns = list(df.columns)
y_column= 'FRAUD_NONFRAUD'
x_columns.remove(y_column)

timestamp_attributes = ['PWD_UPDT_TS', 'PH_NUM_UPDT_TS', 'CUST_SINCE_DT', 'TRAN_DT', 'TRAN_TS', 'ACTVY_DT']
nominal_attributes = [
                      'CARR_NAME',
                      'RGN_NAME',
                      'STATE_PRVNC_TXT',
                      'ALERT_TRGR_CD',
                      'DVC_TYPE_TXT',
                      'AUTHC_PRIM_TYPE_CD',
                      'AUTHC_SCNDRY_STAT_TXT',
                      'CUST_STATE',
                      'ACTN_CD',
                      'ACTN_INTNL_TXT',
                      'TRAN_TYPE_CD',
                      'FRAUD_NONFRAUD']
numeric_attributes = ['TRAN_AMT', 'ACCT_PRE_TRAN_AVAIL_BAL', 'CUST_AGE', 'OPEN_ACCT_CT', 'WF_dvc_age']

# Drop timestamp attributes
df = df.drop(timestamp_attributes, axis = 'columns')
dataframes = dict()

# remove rows with NaN
dataframes['remove_nan'] = df.copy()
dataframes['remove_nan'] = dataframes['remove_nan'].dropna()

# replace NaN with mode
dataframes['nan_with_mode'] = df.copy()
for column in nominal_attributes:
  mean_value=dataframes['nan_with_mode'][column].mode()
  dataframes['nan_with_mode'][column].fillna(value=mean_value, inplace=True)


# One hot encode nominal attributes
for dataframe_type in ['remove_nan', 'nan_with_mode']:
  _df = dataframes[dataframe_type]
  for column in nominal_attributes:
    encoding = pd.get_dummies(_df[column], prefix=column)
    _df = _df.drop(column, axis = 'columns')
    _df = _df.join(encoding)
  dataframes[dataframe_type] = _df


# Standardize and normalize numerical attributes
for dataframe_type in ['remove_nan', 'nan_with_mode']:
  _df_normalize = dataframes[dataframe_type]
  _df_standardize = dataframes[dataframe_type]
  _df_normalize[numeric_attributes] = preprocessing.normalize(_df_normalize[numeric_attributes])
  _df_standardize[numeric_attributes] = preprocessing.normalize(_df_standardize[numeric_attributes])
  dataframes[dataframe_type] = dict()
  dataframes[dataframe_type]['normalized'] = _df_normalize
  dataframes[dataframe_type]['standardized'] = _df_standardize

dataset = dataframes['remove_nan']["normalized"]

X = dataset[dataset.columns[:-2]]
Y = dataset[['FRAUD_NONFRAUD_Fraud']]

x_train = X.iloc[0:int(dataset.shape[0]*0.80)]
y_train = Y.iloc[0:int(dataset.shape[0]*0.80)]

x_test = X.iloc[int(dataset.shape[0]*0.80): int(dataset.shape[0])]
y_test = Y.iloc[int(dataset.shape[0]*0.80): int(dataset.shape[0])]

def LR(x_train,y_train,x_test,y_test): 
  logisticRegr = LogisticRegression()
  logisticRegr.fit(x_train, y_train)
  y_pred = logisticRegr.predict(x_test)
  train_acc =  metrics.accuracy_score(y_test, y_pred)
  x_pred = logisticRegr.predict(x_train)
  test_acc = metrics.accuracy_score(y_train, x_pred)
  model = logisticRegr.get_params

  return model, train_acc , test_acc

LR(x_train,y_train,x_test,y_test)
