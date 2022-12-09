# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:59:15 2022

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("archive (1).zip")
df
df.head(10)
df.shape
df.isnull().sum()
df.info
df.dtypes
df.isnull
df.describe()
x=df.iloc[:,[0,1]].values
x
y=df.iloc[:,2].values
y
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(x)
x
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)
x_train
x_test
y_train
y_test
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression()
lr_model.fit(x_train,y_train)
ypred=lr_model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,ypred)
cm

from matplotlib.colors import ListedColormap  
x_set, y_set = x_train, y_train  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01), 
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, lr_model.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
plt.title('Logistic Regression (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  
