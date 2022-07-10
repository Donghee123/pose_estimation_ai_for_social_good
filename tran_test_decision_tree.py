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

## 데이터 표준화
#sc = StandardScaler()
#sc.fit(x_train)
#x_train_std = sc.transform(x_train)
#x_test_std = sc.transform(x_test)

tree_model = DecisionTreeClassifier()
tree_model.fit(x_train, y_train)

y_pred = tree_model.predict(x_test)

print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]))
print(accuracy_score(y_test, y_pred))
    
    
