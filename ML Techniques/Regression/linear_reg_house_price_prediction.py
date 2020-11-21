# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:47:50 2020

@author: XS653RB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

###single variate scenario
# initialize list of lists 
data = [[2600, 550000], [3000,565000], [3200,610000], [3600,680000], [4000,725000]] 
  
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['area', 'price']) 
plt.xlabel('area(area(sqr ft))')
plt.ylabel('price(US$)')
plt.scatter(df.area,df.price, color = 'red', marker = '+')
reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
m = reg.coef_
c = reg.intercept_
# plt.scatter(df.area,m*df.area + c, color = 'blue')

##training data plot
plt.plot(df.area,df.price,'r*')
##plotting the  line with minimum error
plt.plot(df.area,m*df.area + c,'g')
plt.show()




###multiple variate scenario
data = [[2600,3,20,550000], [3000,4,15,565000], [3200,np.nan,18,610000], [3600,3,30,680000], [4000,5,8,725000]] 
df = pd.DataFrame(data, columns = ['area','bedrooms','age','price']) 
df.dropna(inplace = True)
# median_br = math.floor(df.bedrooms.median())
# df.bedrooms.fillna(median_br,inplace = True)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)