import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def randomforest(X_train, Y_train, X_test, Y_test):

      model = RandomForestClassifier(bootstrap=True,criterion='gini',
                       max_depth=20, max_features='auto',min_impurity_decrease=0.0,
                       min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,
                       n_estimators=100,random_state=46,warm_start=True)

      model.fit(X_train,Y_train)
      
      return round(accuracy_score(model.predict(X_train),Y_train),2),round(accuracy_score(model.predict(X_test),Y_test),2),model.get_params()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="randomforest.py")
    parser.add_argument("-filename", default="data",help="Path to Data")
    args = parser.parse_args()
    randomforest(args.filename)