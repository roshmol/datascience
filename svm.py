# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:22:58 2022

@author: USER
"""

import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

df=pd.read_csv("archive (1).zip")
df.shape
df
x=df.iloc[:,0:2].values
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
#creat confusion matrix
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,ypred)
cm
model.score(x_test,y_test)
cr=classification_report(y_test,ypred)
cr
ac=accuracy_score(y_test,ypred)
ac
from matplotlib.colors import ListedColormap  
x_set, y_set = x_train, y_train  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, model.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
plt.title('SVM (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()
