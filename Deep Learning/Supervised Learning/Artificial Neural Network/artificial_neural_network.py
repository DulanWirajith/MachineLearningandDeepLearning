import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data set
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# categorical data encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHotEncoder",  # Just a name
         OneHotEncoder(),  # The transformer class
         [1]  # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'
)
X = transformer.fit_transform(X)

# Avoiding Dummy Variable Trap
X = X[:, 1:]

transformer = ColumnTransformer(
    transformers=[
        ("OneHotEncoder",  # Just a name
         OneHotEncoder(),  # The transformer class
         [3]  # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'
)
X = transformer.fit_transform(X)

# Avoiding Dummy Variable Trap
X = X[:, 1:]

# feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# spliting the dataset into test data and training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create ANN Model and fit into dataset
import keras
from keras.models import Sequential
from keras.layers import Dense

#     Initializing ANN
classifier_ann = Sequential()

#    Adding the input layer and first hidden layer
classifier_ann.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))

#    Adding the second hidden layer
classifier_ann.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

#    Adding the output layer
classifier_ann.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#    Compiling ANN
classifier_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#   Fitting the ANN to Training Set
classifier_ann.fit(X_train, y_train, batch_size=10, epochs=100)

# Predict using ANN Model
y_pred = classifier_ann.predict(X_test)

# converting probabilities to prediction results
y_pred=(y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
