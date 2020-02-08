## Importing the dataset 

df = read.csv('Ads_CTR_Optimisation.csv')

## Upper Confidence Bound
N = 10000
d = 10
no_of_selections = integer(d)
sum_of_reward = integer(d)
ads_selected = integer(0)
total_reward = 0

for (n in 1:N) {
  max_upper_bound = 0
  ad = 0 
  for(i in 1:d) {
    if(no_of_selections[i] > 0) {
      avg = sum_of_reward[i]/no_of_selections[i]
      delta = sqrt(1.5*log(n)/no_of_selections[i])
      ub = avg + delta
    } else {
      ub = 1e400
    }  
    if (ub > max_upper_bound) {
      max_upper_bound = ub
      ad = i
    }  
  }
  ads_selected = append(ads_selected, ad)
  no_of_selections[ad] = no_of_selections[ad] + 1
  reward = df[n,ad]
  sum_of_reward[ad] = sum_of_reward[ad] + reward
  total_reward = total_reward + reward
}



## Thompson Sampling
N = 10000
d = 10
no_of_reward_0 = integer(d)
no_of_reward_1 = integer(d)
ads_selected = integer(0)
total_reward = 0

for (n in 1:N) {
  max_random = 0
  ad = 0 
  for(i in 1:d) {
    random_beta = rbeta(n = 1, shape1 = no_of_reward_1[i] + 1 , shape2 = no_of_reward_0[i] + 1) 
    if (random_beta > max_random) {
      max_random = random_beta
      ad = i
    }  
  }
  
  ads_selected = append(ads_selected, ad)
  reward = df[n,ad]
  if (reward == 1){
    no_of_reward_1[ad] = no_of_reward_1[ad] + 1
  } else {
    no_of_reward_0[ad] = no_of_reward_0[ad] + 1
  }
  total_reward = total_reward + reward
}

## Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Count')
    
    
    
  
