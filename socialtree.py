# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:38:04 2022

@author: USER
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("archive (1).zip")
x=df.iloc[:,0:2].values
x
y=df.iloc[:,-1].values
y
sc=StandardScaler()
X=sc.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion="entropy",random_state=0)

model.fit(x_train,y_train)

from sklearn.tree import export_text
tree= export_text(model,feature_names=['age','salary'])
ypred=model.predict(x_test)
ypred
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,ypred)
model.score(x_test,y_test)
cr=classification_report(y_test,ypred)
ac=accuracy_score(y_test,ypred)
ac
lg_model=LogisticRegression()
lg_model.fit(x_train,y_train)
y_pred=lg_model.predict(x_test)
model.score(x_test,y_test)
cr1=classification_report(y_test,ypred)
ac1=accuracy_score(y_test,ypred)
ac1
#ROC AUC CURVE
from sklearn.metrics import roc_curve,auc,roc_auc_score
fpr,tpr,thresh=roc_curve(y_test,ypred)


a=auc(fpr,tpr)
plt.plot(fpr,tpr,color="green",label=("AUC value: %0.2f"%(a)))
plt.plot([0,1],[0,1],"--",color="red")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC-AUC CURVE")
plt.legend(loc="best")
plt.show()




from sklearn.metrics import roc_auc_score,roc_curve,auc
fpr,tpr,thresh=roc_curve(y_test,ypred)
a = auc(fpr,tpr)


fpr1,tpr1,thresh = roc_curve(y_test,y_pred)
b = auc(fpr1,tpr1)

plt.plot(fpr,tpr,color="green",label=("AUC value of Decision tree: %0.2f"%(a)))
plt.plot(fpr1,tpr1,color="blue",label=("AUC value of logistic Regression: %0.2f"%(b)))
plt.plot([0,1],[0,1],"--",color="red")
plt.xlabel("False positive rate")
plt.ylabel("True Positive rate")
plt.title("ROC-AUC Curve")
plt.legend(loc="best")
plt.show()
