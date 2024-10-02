# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:24:23 2024

@author: Admin
"""
import pandas as pd 
import matplotlib.pyplot as plt
univ1=pd.read_excel("C:/cluster/University_Clustering.xlsx")

a=univ1.describe()
Univ=univ1.drop(['State'],axis=1)

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
 
df_norm=norm_func(Univ.iloc[:,1:])  
b=df_norm.describe() 
 
    
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchica clustering dendogram")
plt.xlabel("index")
plt.ylabel("distance")

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show


from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',metric='euclidean').fit(df_norm)
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
Univ['clust']=cluster_labels


univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]

univ1.iloc[:,2:].groupby(univ1.clust).mean()





