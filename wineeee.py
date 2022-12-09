# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:50:17 2022

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
wine=load_wine()
wine
dir(wine)
wine['feature_names']
wine['data']
wine['target_names']
df=pd.DataFrame(wine["data"],columns=wine["feature_names"])
x=df.data
y=df.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
model=GaussianNB()
model.fit(x_train,y_train)
ypred=model.predict(x_test)
model.score(x_train,y_train)

model1=BernoulliNB()
model1.fit(x_train,y_train)
ypred=model1.predict(x_test)
model1.score(x_train,y_train)

model2=MultinomialNB()
model2.fit(x_train,y_train)
ypred=model2.predict(x_test)
model2.score(x_train,y_train)

