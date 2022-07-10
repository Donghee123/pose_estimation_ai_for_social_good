from xgboost import XGBClassifier
from sklearn import metrics
from tkinter.messagebox import NO
import numpy as np   
import torch
from network import Network
from dataloader import *
import torch.optim as optim
import os

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

## 의사결정트리 모델
from sklearn.tree import DecisionTreeClassifier

## 정확도 예측 모듈(연속적인 데이터에서 사용 안함)
from sklearn.metrics import accuracy_score

## 의사결정트리 시각화
from sklearn.tree import export_graphviz

from sklearn.preprocessing import StandardScaler

def saveModel(PATH, clf_xgb):
    clf_xgb.save_model(PATH)

def loadModel(PATH):
    # define new model and load save parameters
    loaded_clf = XGBClassifier()
    model_path = 'PATH'
    loaded_clf.load_model(model_path)

# If you want to use full Dataset, please pass None to csvpath
strDataFolderPath = os.path.join('sample_image_folder', 'skeleton_npy')

dictOfLabes = {'good' : 0, 'left' : 1, 'right' : 2, 'turtleneck' : 3}
strDataFolderlist = os.listdir(strDataFolderPath)
        
X = []
Y = []

for strLabelFolderPath in strDataFolderlist:
    strOneLabelDataPath = os.path.join(strDataFolderPath, strLabelFolderPath)
    listOfOneLabelDataPath = os.listdir(strOneLabelDataPath)
    for strNPDataPath in listOfOneLabelDataPath:
        X.append(np.load(os.path.join(strOneLabelDataPath,strNPDataPath), allow_pickle=True).flatten())
        Y.append(dictOfLabes[strLabelFolderPath])

train_size = int(0.8 * len(X))
test_size = len(X) - train_size

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=1)

#n_estimators : 트리 갯수
n_estimators = 240

clf_xgb = XGBClassifier(max_depth=8,
    learning_rate=0.1,
    n_estimators=n_estimators,
    verbosity=0,
    silent=None,
    objective="binary:logitraw",
    booster='gbtree',
    n_jobs=0,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0.1,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0.5,
    reg_lambda=1,
    scale_pos_weight=0.01,
    base_score=0.01,
    random_state=0,
    seed=None)

clf_xgb.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            early_stopping_rounds=100,
            verbose=10,
            eval_metric='auc')




y_pred = clf_xgb.predict(x_test)
print('AUC score')
print(metrics.accuracy_score(y_test,y_pred))
print('Confusion matrix score')
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]))
    

"""
y_pred = clf_xgb.predict_proba(X_valid)[:,1]
valid_acc = metrics.roc_auc_score(y_valid,y_pred)
print(valid_acc)

y_pred = clf_xgb.predict_proba(X_test)[:,1]
test_acc = metrics.roc_auc_score(y_test,y_pred)
print(test_acc)
"""