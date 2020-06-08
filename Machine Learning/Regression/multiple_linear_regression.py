import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
"""

# Avoiding Dummy Variable Trap
X = X[:, 1:]  # 1: kiyanne column number 1 indan ithuru tikama
"""
# spliting the dataset into test data and training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""

# feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
"""
"""
# Fitting simple linear regression model to training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictin the X_Test values
y_pred = regressor.predict(X_test)
"""

# import statsmodels.regression.linear_model as sm

X = np.append(arr=np.ones(shape=(50, 1), dtype=int).astype(dtype=int), values=X, axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Optimize model using Backward Elimination Method
import statsmodels.api as sm

X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()

# Predictin the X_Test values
for_Test = X_test[:, [0, 3, 5]]
y_pred = regressor_OLS.predict(for_Test)
