import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

def NBC(x_train,y_train,x_test,y_test):
  model = GaussianNB()
  model.fit(x_train, y_train)
  modelf = model.get_params()
  y_pred = model.predict(x_test)
  train_acc = metrics.accuracy_score(y_test, y_pred)
  x_pred = model.predict(x_train)
  test_acc =  metrics.accuracy_score(y_train, x_pred)

  return round(train_acc,2), round(test_acc,2), modelf



