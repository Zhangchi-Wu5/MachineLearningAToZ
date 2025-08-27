#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset = pd.read_csv('datasets/Data.csv')

print("datasets info")
print(dataset)

#create two entities
X = dataset.iloc[:, :-1].values # matrix of features
y = dataset.iloc[:, -1].values  # dependent variable vector (sth want to predict)

print("Featrue Matrix X Info")
print(X)
print ("Dependent Variable y Info")
print(y)