import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

"""
# spliting the dataset into test data and training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""

# feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_Y.fit_transform(y)
# y = sc_Y.fit_transform(np.array(y).reshape(1, -1))
# Z=sc_Y.inverse_transform(y)

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y.ravel())

# Predicting a new result
vec = 6.5
vec = np.array(vec).reshape(1, -1)
y_pred = regressor.predict(sc_X.transform(vec))
y_pred = sc_Y.inverse_transform(y_pred)
y_pred = y_pred.ravel()

# # Visualising the Regression results
# plt.scatter(X, y, color='red')
# plt.plot(X, regressor.predict(X), color='blue')
# plt.title('Truth or Bluff (Regression Model)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
