import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn import metrics

def SVM(x_train,y_train,x_test,y_test):
  classifier = svm.SVC(kernel='sigmoid') 
  classifier.fit(x_train,y_train)
  y_pred = classifier.predict(x_test)
  train_acc = metrics.accuracy_score(y_test, y_pred)
  x_pred = classifier.predict(x_train)
  test_acc = metrics.accuracy_score(y_train, x_pred)
  model = classifier.get_params()
  
  return round(train_acc,2),round(test_acc,2),model

