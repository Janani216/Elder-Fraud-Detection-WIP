import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from lr_fraud_prediction  import LR
from svm_fraud_prediction import SVM
from nbc_fraud_prediction import NBC
from randomforest import randomforest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
dataset = pd.read_csv('data_full.csv',index_col = 0)
columns = dataset.columns[:-2]
#Data Split Train: 80, Test: 20
X =  dataset[columns]
Y = dataset[["FRAUD_NONFRAUD_Fraud","FRAUD_NONFRAUD_Non-Fraud"]]

X_train = X.iloc[0:int(dataset.shape[0]*0.80)]
Y_train = Y[0:int(dataset.shape[0]*0.80)]

X_test = X.iloc[int(dataset.shape[0]*0.80):int(dataset.shape[0])]
Y_test = Y[int(dataset.shape[0]*0.80):int(dataset.shape[0])]


k_fold_dict = {}
Train_NBC_accu_dict = {}

accuracy_dict = {}
bucket = 0

for i in range(0,int(len(dataset)),int(len(dataset)//10)):
    k_fold_dict[bucket] = pd.DataFrame(dataset.iloc[int(i):int(i+len(dataset)//10)])
    bucket = bucket + 1

accs_RF  = {}
accs_LR  = {}
accs_NB  = {}
accs_SVM = {}
for j in range(0,10):
    train_set = pd.DataFrame()
    test_set = pd.DataFrame() 
    test_set = test_set.append(k_fold_dict[j])
    for k in range(0,10):
        if j != k:
            train_set = train_set.append(k_fold_dict[k])

    X_train = train_set.iloc[0:int(len(train_set)*0.80),:-2]
    Y_train = train_set.iloc[0:int(len(train_set)*0.80),-2:]
    Y_train_svm = train_set.iloc[0:int(len(train_set)*0.80),-1:]

    X_test = train_set.iloc[int(len(train_set)*0.80):int(len(train_set)),:-2]
    Y_test = train_set.iloc[int(len(train_set)*0.80):int(len(train_set)),-2:]
    Y_test_svm = train_set.iloc[int(len(train_set)*0.80):int(len(train_set)),-1:]

    train_acc, test_acc, model = randomforest(X_train,Y_train,X_test,Y_test)
    accs_RF[j] = [train_acc, test_acc,model]

    train_acc, test_acc, model = LR(X_train,Y_train_svm,X_test,Y_test_svm)
    accs_LR[j] = [train_acc, test_acc,model]

    train_acc, test_acc, model = NBC(X_train,Y_train_svm,X_test,Y_test_svm)
    accs_NB[j] = [train_acc, test_acc,model]

    train_acc, test_acc, model = SVM(X_train,Y_train_svm,X_test,Y_test_svm)
    accs_SVM[j] = [train_acc, test_acc,model]

accuracy_dict["randomforest"]           = accs_RF
accuracy_dict["Logistic Regression"]    = accs_LR
accuracy_dict["Naive Bayes"]            = accs_NB
accuracy_dict["Support Vector Machine"] = accs_SVM

models = accuracy_dict.keys()
a_file = open("accuracy.pkl", "wb")
pickle.dump(accuracy_dict, a_file)
a_file.close()

for model in models:
    train_acc = 0
    test_acc  = 0
    val_max = 0
    model_best_temp = []
    model_best = {}
    for idx in range(0,10):
        train_acc += accuracy_dict[model][idx][0]
        test_acc += accuracy_dict[model][idx][1]
        if val_max > accuracy_dict[model][idx][1]:
            val_max = accuracy_dict[model][idx][1]
            model_best_temp = accuracy_dict[model][idx][2]
    model_best[model] = model_best_temp
    print("For ",model)
    print("Mean Train acc: ",round(train_acc/10,2))
    print("Mean Test acc: ",round(test_acc/10,2))
