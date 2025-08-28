# 第4步：处理缺失数据
# 这个脚本展示了如何处理数据集中的缺失值

import pandas as pd   # 用于数据处理
import numpy as np    # 用于数值计算

# 导入数据集
dataset = pd.read_csv('datasets/Data.csv')

print("原始数据集信息:")
print(dataset)

# 使用 SimpleImputer 类来处理缺失数据
# SimpleImputer是scikit-learn中用于填充缺失值的工具
# 我们将创建一个SimpleImputer类的实例（对象）
# 这个对象将允许我们用平均值来替换缺失的数据
from sklearn.impute import SimpleImputer

# 创建填充器对象：
# missing_values=np.nan：指定要替换的缺失值类型
# strategy='mean'：使用平均值策略来填充缺失值
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

print("填充器对象信息:")
print(imputer)

# 提取特征矩阵和目标变量
X = dataset.iloc[:, :-1].values  # 特征矩阵（所有列除了最后一列）
y = dataset.iloc[:, -1].values   # 目标变量（最后一列）

print("处理前的特征矩阵 X:")
print(X)
print("目标变量 y:")
print(y)

# 对特征矩阵的第1列和第2列（年龄和薪资）应用填充
# 这两列包含数值型数据，可能有缺失值
imputer.fit(X[:, 1:3])  # 学习这两列的统计特征（平均值）
X[:, 1:3] = imputer.transform(X[:, 1:3])  # 应用转换，用平均值替换缺失值

print("填充缺失值后的特征矩阵 X:")
print(X)