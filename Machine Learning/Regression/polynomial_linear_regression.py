import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""
# import data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# categorical data encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHotEncoder",  # Just a name
         OneHotEncoder(),  # The transformer class
         [3]  # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'
)
dummy_encoded_X = transformer.fit_transform(X)
X = dummy_encoded_X.astype(float)

# Avoiding Dummy Variable Trap
X = X[:, 1:]
"""

# Fitting Polynomial Regression Model to dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree=10)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Predicting new result using Polynomial Regression
vec = 6.5
vec = np.array(vec).reshape(1, -1)
y_pred = lin_reg.predict(poly_reg.fit_transform(vec))

# Visualizing the Test Set Results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X_poly), color='blue')
# plt.plot(X, poly_reg.predict(X_poly), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.get_backend()
plt.show()

# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color='red')
# plt.plot(X_grid, lin_reg.predict(X_grid), color='blue')
# plt.title('Truth or Bluff (Regression Model)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()
