# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:27:25 2022

@author: USER
"""

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
df=load_breast_cancer()
x=df.data
y=df.target
x.shape
y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
model=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
model.fit(x_train,y_train)
ypred=model.predict(x_test)
ypred


cm=confusion_matrix(y_test,ypred)
cm
model.score(x_test,y_test)
ac=accuracy_score(y_test,ypred)
ac
