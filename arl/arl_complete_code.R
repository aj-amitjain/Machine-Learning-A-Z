

## ARL Complete Code

## Importing the dataset
#install.packages('arules')
library(arules)
df = read.transactions('Market_Basket_Optimisation.csv', header = F, sep =  ',', rm.duplicates = T)
summary(df)
itemFrequencyPlot(df, topN = 10)

## Making rules with apriori

rules = apriori(df, parameter = list(support = 0.003, confidence = 0.4))

inspect(sort(rules, by='lift')[1:10])

#------------------------- OR ----------------------------#

## Making sets of associated items with Eclat 

sets = eclat(df, parameter = list(support = 0.003, minlen=2))

inspect(sort(sets, by='support')[1:10]) 


