import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math

d = 10
N = 10000
number_of_times_selections = [0] * d
sum_of_rewards = [0] * d
selected_ads = []
total_rewards = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if number_of_times_selections[i] > 10:
            average_reward = sum_of_rewards[i] / number_of_times_selections[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / number_of_times_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    selected_ads.append(ad)
    number_of_times_selections[ad] = number_of_times_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_rewards += reward

# Visualising the results
plt.hist(selected_ads)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times ad selected')
plt.show()
