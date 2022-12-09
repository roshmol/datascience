# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 20:21:57 2022

@author: USER
"""

import pandas as pd
df=pd.read_csv("Wine.csv")
df
x=df.iloc[:,0:13]
x
y=df.iloc[:,-1]
y
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(x)
X
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression()
lr_model.fit(x_train,y_train)
ypred=lr_model.predict(x_test)
ypred
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,ypred)
cm
lr_model.score(x_test,y_test)
cr=classification_report(y_test,ypred)
cr
ac=accuracy_score(y_test,ypred)
ac
