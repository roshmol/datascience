# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:04:00 2022

@author: USER
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC #support vector machine
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
df=pd.read_csv("Wine.csv")
df.shape
df
x=df.iloc[:,0:13].values
x
y=df.iloc[:,-1].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
model=SVC(kernel='linear',random_state=0)
model.fit(x_train,y_train)
ypred=model.predict(x_test)
ypred
cm=confusion_matrix(y_test,ypred)
cm
model.score(x_test,y_test)
cr=classification_report(y_test,ypred)
cr
ac=accuracy_score(y_test,ypred)
ac
