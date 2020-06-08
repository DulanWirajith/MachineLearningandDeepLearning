import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('GOOG.csv', date_parser=True)

data_training = data[data['Date'] < '2019-01-01'].copy()
data_test = data[data['Date'] >= '2019-01-01'].copy()

data_training = data_training.drop(['Date', 'Adj Close'], axis=1)
data_training = np.array(data_training)

scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)

X_train = []
y_train = []

for i in range(60, data_training.shape[0]):
    X_train.append(data_training[i - 60:i])
    y_train.append(data_training[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
