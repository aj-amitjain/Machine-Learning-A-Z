# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:12:07 2020

@author: Amit Anchalia

@Description: Clustering Complete Code
"""

## Importing the library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing the dataset
df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:, 3:5]

## Fitting the Clusters

#-------------------------------------------------------#

## K-Mean

## Finding optimal K (no of cluster) using elbow method (reduce WCSS - Within Cluster Sum of Square) 
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')
plt.show

from sklearn.cluster import KMeans
title = 'K-Means'
cluster = KMeans(n_clusters = 5, init='k-means++', random_state=0)
y_pred  = cluster.fit_predict(X)

#------------------------- OR ----------------------------#

## Hierarchical Clustering

## Finding optimal no of cluster using dendrogram method (reduce the variance within clusters)
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
title = 'Hierarchical Clustering'
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred  = cluster.fit_predict(X)

## Visualizing the clusters

#-------------------------------------------------------#
title_ = title + ' (Clusters of Customers)' 
colors = ['red', 'green', 'blue', 'cyan', 'orange']
for i in range(0,5):
    label_ = 'cluster' + str(i)
    plt.scatter(X.iloc[y_pred == i, 0], X.iloc[y_pred == i, 1], s=10, c=colors[i], label=label_)

plt.title(title_)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()