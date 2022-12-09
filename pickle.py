# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:47:52 2022

@author: USER
"""

import pandas as pd
import pickle
f1=open(file="navebasemodel.pkl",mode="br")
m1=pickle.load(f1)
f1.close()
f2=open(file="standardscler.pkl",mode="br")
m2=pickle.load(f2)
f1.close()
def prediction(a,b):
    data={'age':a,'estimatedsalary':b}
    df=pd.DataFrame(data,index=[0])
    df=m2.transform(df)
    pred=m1.predict(df)
    if int(pred)==1:
        return 'purchased'
    else:
        return 'no purchased'
age=int(input("enter your age"))  
salary=int(input("enter your salary")) 
prediction(age,salary) 
