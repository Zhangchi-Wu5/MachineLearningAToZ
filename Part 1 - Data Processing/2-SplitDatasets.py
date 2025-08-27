import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('datasets/Data.csv')

print("datasets info")
print(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



#Spliting Training Datasets And Test Datasets

from sklearn.model_selection import train_test_split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 