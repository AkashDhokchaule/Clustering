# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:34:13 2024

@author: Akash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

##generates 2 random numbers in range 0 to 50
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
##create empty dataframe
df_xy = pd.DataFrame(columns=['X', 'Y'])
#assign values to the x and y to these columns 
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x = 'X', y='Y', kind = 'scatter')
model1 = KMeans(n_clusters=3).fit(df_xy)

model1.labels_
df_xy.plot(x = 'X', y = 'Y', c = model1.labels_, kind = 'Scatter', s = 10, cmap = plt.cm.coolwarm)

##########################################################################
Univ1 = pd.read_excel("C:/cluster/University_Clustering.xlsx")
Univ1.describe()
Univ = Univ1.drop(['State'], axis = 1)

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

##apply normalization fuction
df_norm = norm_func(Univ.iloc[:,1:])
'''
what will be the ideal cluster number, will it be 1, 2 or 3
'''

TWSS =[]
k = list(range(2,8))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)##total within sum of square

''''
KMeans inertia also kinow as sum of the distance of all points 
within a cluster from centroid of the poin
'''


TWSS
##as value increases the TWSS value decreses

plt.plot(k, TWSS, 'ro-');
plt.xlabel('No_of_clusters');
plt.ylabel('Total_within_SS')

'''
how to select value of k from elbow curve
when k change from 2 to 3 then decrease
in twss ishiger than 
when k changes from 3 to 4
when k values changes from 5 to 6 decrease
'''

model = KMeans(n_clusters = 3)
model.fit(df_norm)
model.labels_
mb = pd.Series(model.labels_)

Univ['clust'] = mb
Univ.head()
Univ = Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()

Univ.to_csv('kmeans_University.csv', encoding = 'utf-8')
import os
os.getcwd()





