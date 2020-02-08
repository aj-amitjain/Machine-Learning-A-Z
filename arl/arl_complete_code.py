# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:15:02 2020

@author: Amit Anchalia

@Description: ARL
"""

## Importing the library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing the dataset
df = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transcations = []
for i in range(0,7501):
    transcations.append([str(df.values[i,j]) for j in range(0,20)])
    
from apyori import apriori
rules = apriori(transcations, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
print(results) 

