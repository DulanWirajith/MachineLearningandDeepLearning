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

#  FOR RUN KERAS IN TENSORFLOW CPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# making ANN using KerasClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


def build_classifier(optimizer='adam'):
    classifier_ann = Sequential()
    classifier_ann.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
    classifier_ann.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier_ann.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier_ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier_ann


classifier_ann = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)

# evaluating accuracy using k-gold cross validation
from sklearn.model_selection import cross_val_score

import multiprocessing

n_jobs = multiprocessing.cpu_count() - 1

classifier_anns = cross_val_score(estimator=classifier_ann, X=X_train, y=y_train, cv=10, n_jobs=-1)
accuracy = classifier_anns.mean()
variance = classifier_anns.std()
