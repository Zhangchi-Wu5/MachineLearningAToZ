import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('datasets/Data.csv')

print("datasets info")
print(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("X Info")
print(X)
print ("y Info")
print(y)


#Spliting Training Datasets And Test Datasets

from sklearn.model_selection import train_test_split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("X_train Info")
print(X_train)
print("X_test Info")
print(X_test)
print("y_train Info")
print(y_train)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_train[:, 1:3])
X_train[:, 1:3] = imputer.transform(X_train[:, 1:3])
X_test[:, 1:3] = imputer.transform(X_test[:, 1:3])

print("X_train after imputation:")
print(X_train)
print("X_test after imputation:")
print(X_test)