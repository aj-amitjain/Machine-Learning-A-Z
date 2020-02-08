# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:14:51 2020

@author: Amit Anchalia
"""

 ## Importing the library
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

## Importing the dataset 
df = pd.read_csv('Ads_CTR_Optimisation.csv')

## Upper Confidence Bound
d = 10
N = 10000
no_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
ads_selected = []
for n in range(0,N):
    max_ub = 0
    ad = 0
    for i in range(0,d):
        if(no_of_selections[i] > 0):
            avg = sum_of_rewards[i]/no_of_selections[i]
            delta = math.sqrt(1.5 * math.log(n +1)/no_of_selections[i])
            ub = delta + avg
        else:
            ub = 1e400  

        if(ub > max_ub):
            max_ub = ub
            ad = i
   
    ads_selected.append(ad)
    no_of_selections[ad] = no_of_selections[ad] + 1
    reward = df.iloc[n, ad]        
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward

## Thompson Sampling 
d = 10
N = 10000
no_of_reward_0 = [0] * d
no_of_reward_1 = [0] * d
total_reward = 0
ads_selected = []
for n in range(0,N):
    max_random = 0
    ad = 0
    for i in range(0,d):
        random_beta = random.betavariate(no_of_reward_1[i] + 1, no_of_reward_0[i] + 1)
        if(random_beta > max_random):
            max_random = random_beta
            ad = i
   
    ads_selected.append(ad)
    reward = df.iloc[n, ad]        
    if (reward == 1):
        no_of_reward_1[ad] = no_of_reward_1[ad] + 1
    if (reward == 0):
        no_of_reward_0[ad] = no_of_reward_0[ad] + 1
        
    total_reward = total_reward + reward


## Visualizing the ads selected
plt.hist(ads_selected)
plt.title('Histogram for ads Selected')    
plt.xlabel('Ads')                
plt.ylabel('Count')
plt.show()