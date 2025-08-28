import pandas as pd
import numpy as np
#importing datasets
dataset = pd.read_csv('datasets/Data.csv')

print("datasets info")
print(dataset)

# SimpleImputer class
# 然后我们将创建一个实例, SimpleImputer类的一个对象.
# 这个对象将允许我们用平均工资来替换缺失的工资,
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

print("imputer info")
print(imputer)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("X info")
print(X)
print("y info")
print(y)

imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print("Transformed dataset info")
print(X)