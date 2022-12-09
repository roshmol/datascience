# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:28:25 2022

@author: USER
"""

import pandas as pd
import numpy as np
df=pd.read_csv("salaries (4).csv")

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df['company']=label.fit_transform(df.company)
df['job']=label.fit_transform(df.job)
df['degree']=label.fit_transform(df.degree)
x=df.iloc[:,0:3].values
y=df.iloc[:,-1].values
#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

model.fit(x,y)
model.score(x,y)


from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()
model1.fit(x,y)
model1.score(x,y)
