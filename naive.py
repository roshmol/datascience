# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:38:32 2022

@author: USER
"""

import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
df=pd.read_csv("archive (1).zip")
df.shape

x=df.iloc[:,0:2].values

y=df.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(x)
X
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()

model.fit(x_train,y_train)
ypred=model.predict(x_test)
ypred
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,ypred)
cm
model.score(x_test,y_test)
cr=classification_report(y_test,ypred)
cr
ac=accuracy_score(y_test,ypred)
ac
#Model Saving
import pickle 
f1=open(file="navebasemodel.pkl",mode="bw")
pickle.dump(model,f1)
f1.close()
f2=open(file="standardscler.pkl",mode="bw")
pickle.dump(sc,f2)
f2.close()
