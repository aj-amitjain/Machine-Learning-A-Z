
## Classification Complete Code

## Importing the dataset
df = read.csv('Social_Network_Ads.csv')
df = df[3:5]

## Encoding the target feature as factor
df$Purchased = factor(df$Purchased, levels = c(0, 1))

## Splitting the dataset into the Train set and Test set
# install.packages('caTools') ## In case you don't have the library installed
library(caTools)
set.seed(123)
split = sample.split(df$Purchased, SplitRatio = 0.75)
train_set = subset(df, split == T)
test_set = subset(df, split == F)

## Feature Scaling
train_set[-3] = scale(train_set[-3])
test_set[-3] = scale(test_set[-3])

## Fitting the Classifiers 

#-------------------------------------------------------# 

## Logistic Regressor
title = 'Logistic Regressor'
classifier = glm(formula = Purchased ~ . ,
                 family = binomial,
                 data = train_set)


prob_pred = predict(classifier,  type='response', newdata = test_set[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)                 

#------------------------- OR ----------------------------#

## KNN
# install.package('class')
library(class)
title = 'KNN'
y_pred = knn(train = train_set[-3],
             test = test_set[-3],
             cl = train_set[, 3],
             k = 5,
             prob = T)

#------------------------- OR ----------------------------#

## SVM 
#install.packages('e1070')
library(e1071)
title = 'SVM'
classifier = svm(formula = Purchased ~ ., 
                 data = train_set,
                 type = 'C-classification',
                 kernel = 'linear')

y_pred = predict(classifier, newdata = test_set[-3])

#------------------------- OR ----------------------------#  

## Kernel SVM : It's SVM with non-linear kernel 
#install.packages('e1070')
library(e1071)
title = 'Kernel SVM'
classifier = svm(formula = Purchased ~ ., 
                 data = train_set,
                 type = 'C-classification',
                 kernel = 'radial')

y_pred = predict(classifier, newdata = test_set[-3])

#------------------------- OR ----------------------------#

## Naive Bayes 
#install.packages('e1070')
library(e1071)
title = 'Navie Bayes'
classifier = naiveBayes(formula = Purchased ~ ., 
                        data = train_set)

y_pred = predict(classifier, newdata = test_set[-3])

#------------------------- OR ----------------------------#

## Decision Tree 
#install.packages('rpart')
library(rpart)
title = 'Decision Tree'
classifier = rpart(formula = Purchased ~ ., 
                 data = train_set)

y_pred = predict(classifier, type='class', newdata = test_set[-3])

#-------------------------------------------------------#

## Random Forest  
#install.packages('randomForest')
library(randomForest)
title = 'Random Forest'
classifier = randomForest(x=train_set[, -3],
                          y=train_set[, 3],
                          ntree = 50)

## Predicting the Test set results for Randon Forest Classifier 
y_pred = predict(classifier, newdata = test_set[-3])

#-------------------------------------------------------#

## Making the Confusion Matrix for the classifier 
cm = table(test_set[, 3], y_pred)
print(cm)

## Visualising the results for classifers,
## Just run the section of any classifier and 
## then run the below code as per the classifier too visualise the Model plot for that classifier.

## Training set results
#install.packages("ElemStatLearn")  ## In case you don't have the library installed
library(ElemStatLearn)
set = train_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid = expand.grid(X1, X2)
colnames(grid) = c('Age', 'EstimatedSalary')

## Predicting grid for each classifier

#-------------------------------------------------------#

## Logistic regression
prob_grid_pred = predict(classifier, type='response', newdata = grid)
grid_pred = ifelse(prob_grid_pred > 0.5, 1, 0)

#------------------------- OR ----------------------------#

## KNN
grid_pred = knn(train = train_set[-3],
                test = grid,
                cl = train_set[, 3],
                k = 5,
                prob = T)

#------------------------- OR ----------------------------#

## SVM Or Kernal SVM Or Naive Bayes Or Random Forest 
grid_pred = predict(classifier, newdata = grid)

#------------------------- OR ----------------------------#

## Decision Tree
grid_pred = predict(classifier, type='class', newdata = grid)

#-------------------------------------------------------#

plot(set[, -3],
     main = paste(title, '(Train set)', sep = " "),
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(grid_pred), length(X1), length(X2)), add = T)
points(grid, pch = '.', col = ifelse(grid_pred == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

## Test set results
#library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid = expand.grid(X1, X2)
colnames(grid) = c('Age', 'EstimatedSalary')

## Predicting grid for each classifier

#-------------------------------------------------------#

## Logistic regression
prob_grid_pred = predict(classifier, type='response', newdata = grid)
grid_pred = ifelse(prob_grid_pred > 0.5, 1, 0)

#------------------------- OR ----------------------------#

## KNN
grid_pred = knn(train = train_set[-3],
                test = grid,
                cl = train_set[, 3],
                k = 5,
                prob = T)

#------------------------- OR ----------------------------#

## SVM Or Kernal SVM Or Naive Bayes Or Random Forest 
grid_pred = predict(classifier, newdata = grid)

#------------------------- OR ----------------------------#

## Decision Tree
grid_pred = predict(classifier, type='class', newdata = grid)

#-------------------------------------------------------#

plot(set[, -3],
     main = paste(title, '(Test set)', sep = " "),
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(grid_pred), length(X1), length(X2)), add = T)
points(grid, pch = '.', col = ifelse(grid_pred == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

