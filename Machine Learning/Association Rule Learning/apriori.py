import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Imorting dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transcations = []
trans = []
for i in range(0,7501):
    transcations.append([str(dataset.values[i,j]) for j in range(0,20)])
# for i in range(0, 7501):
#     for j in range(0, 20):
#         trans.append(str(dataset.values[i, j]))
#     transcations.append(trans)

# Traning Apriori on data set
from apyori import apriori

rules = apriori(transcations, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# visualising rules
results = list(rules)
