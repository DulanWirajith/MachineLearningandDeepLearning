import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import random

d = 10
N = 10000
number_of_rewards_1_for_rach_ad = [0] * d
number_of_rewards_0_for_rach_ad = [0] * d
selected_ads = []
total_rewards = 0
for n in range(0, N):
    max_random_theta = 0
    ad = 0
    for i in range(0, d):
        random_theta = random.betavariate(number_of_rewards_1_for_rach_ad[i] + 1, number_of_rewards_0_for_rach_ad[i] + 1)
        if random_theta > max_random_theta:
            max_random_theta = random_theta
            ad = i
    selected_ads.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1_for_rach_ad[ad] += 1
    else:
        number_of_rewards_0_for_rach_ad[ad] += 1
    total_rewards += reward

# Visualising the results
plt.hist(selected_ads)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times ad selected')
plt.show()
