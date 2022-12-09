# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 22:27:04 2022

@author: USER
"""

import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
df=pd.read_csv("Wine.csv")
df.shape
df
x=df.iloc[:,0:13].values
x
y=df.iloc[:,-1].values
y
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(x)
X
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=4,metric='minkowski',p=2)
acc=[]
model.fit(x_train,y_train)
#predicting the test
ypred=model.predict(x_test)
ypred
#creat confusion matrix
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,ypred)
cm
model.score(x_test,y_test)
cr=classification_report(y_test,ypred)
cr
ac=accuracy_score(y_test,ypred)
ac
