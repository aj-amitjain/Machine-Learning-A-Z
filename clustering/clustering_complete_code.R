

## Clustering Complete Code

## Importing the dataset
df = read.csv('Mall_Customers.csv')
df = df[4:5]

## Fitting the Clusters 

#-------------------------------------------------------# 

## K-Means

## Finding optimal K (no of cluster) using elbow method (reduce WCSS - Within Cluster Sum of Square) 
title = 'K-Means'
library(class)
set.seed(123)
wcss = vector()
for (i in 1:10)
  wcss[i] = sum(kmeans(df,i)$withinss)

plot(c(1:10), wcss, 
     type='b',
     main= 'Elbow Method',
     xlab = 'No of Clusters',
     ylab = 'WCSS')

set.seed(1234)
clust = kmeans(df, 5)
y_pred = clust$cluster


#------------------------- OR ----------------------------#

## Hierarchical Clustering 

## Finding optimal no of cluster using dendrogram method (reduce the variance within clusters)

title = 'Hierarchical Clustering'
dendrogram = hclust(d = dist(df, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = 'Dendrogram',
     xlab = 'Customers',
     ylab = 'Euclidean Distances')


hc = hclust(d = dist(df, method = 'euclidean'), method = 'ward.D')
y_pred = cutree(hc, 5)

#-------------------------------------------------------#

## Visualising the results for clusters,

## K-Means Or Hierarchichal Clustering

library(cluster)
clusplot(df,
         y_pred,
         lines=0,
         color=T,
         labels=4,
         plotchar = F,
         span = T,
         main = paste('Cluster Plot (' , title , ')'),
         xlab = 'Annual Income (k$)',
         ylab = 'Spending Score (1-100)')

