# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:35:41 2022

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits=load_digits()
dir(digits)
digits['data']
digits['feature_names']
df=pd.DataFrame(digits["data"],columns=digits["feature_names"])
x=digits.data
y=digits.target
digits['data'][500]
digits.target[500]
plt.matshow(digits.images[500])
digits.data[11]
digits.images[11]
digits.target[11]
plt.matshow(digits.images[11])
plt.colormaps()
plt.set_cmap('tab20c_r')
