import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def LR(x_train,y_train,x_test,y_test): 
  logisticRegr = LogisticRegression()
  logisticRegr.fit(x_train, y_train)
  y_pred = logisticRegr.predict(x_test)
  train_acc =  metrics.accuracy_score(y_test, y_pred)
  x_pred = logisticRegr.predict(x_train)
  test_acc = metrics.accuracy_score(y_train, x_pred)
  model = logisticRegr.get_params

  return round(train_acc,2) , round(test_acc,2), model


