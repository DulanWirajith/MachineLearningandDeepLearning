import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# add missing values
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=0)
imputerFit = imputer.fit(X[:, [1, 2]])
X[:, [1, 2]] = imputerFit.transform(X[:, [1, 2]])

# categorical data encoding
from sklearn.preprocessing import OneHotEncoder
# onehotencoder = OneHotEncoder(categorical_features=[0])
# asdfX=onehotencoder.fit_transform(X).toarray()
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHotEncoder",  # Just a name
         OneHotEncoder(),  # The transformer class
         [0]  # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'
)
dummy_encoded_X = transformer.fit_transform(X)
X = dummy_encoded_X

from sklearn.preprocessing import LabelEncoder

labelEncoder_Y = LabelEncoder()
normally_encoded_Y = labelEncoder_Y.fit_transform(y)
y = normally_encoded_Y

# spliting the dataset into test data and training data
# from sklearn.cross_validation import train_set_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_Y = StandardScaler()
y_train = sc_Y.fit_transform([y_train])

Z = sc_Y.inverse_transform(y_train)
