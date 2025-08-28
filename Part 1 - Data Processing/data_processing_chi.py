# 数据预处理工具 - 完整版本
# 这个脚本演示了机器学习中数据预处理的主要步骤

# 导入必要的库
import numpy as np           # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import pandas as pd          # 用于数据操作和分析

# 导入数据集
dataset = pd.read_csv('datasets/Data.csv')

print("数据集信息:")
print(dataset)

# 创建特征矩阵X和目标变量y
X = dataset.iloc[:, :-1].values  # 特征矩阵：取除最后一列外的所有列
y = dataset.iloc[:, -1].values   # 目标变量：取最后一列

print("特征矩阵 X 的信息:")
print(X)
print("目标变量 y 的信息:")
print(y)

# 将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split

# test_size=0.2 表示测试集占20%，训练集占80%
# random_state=1 设置随机种子，确保结果可重现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("训练集特征 X_train 的信息:")
print(X_train)
print("测试集特征 X_test 的信息:")
print(X_test)
print("训练集目标变量 y_train 的信息:")
print(y_train)

# 处理缺失数据
# 使用SimpleImputer类来替换缺失值
from sklearn.impute import SimpleImputer

# 创建填充器对象：将缺失值（NaN）用平均值替换
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 对训练集的第1和第2列（年龄和薪水）进行拟合
imputer.fit(X_train[:, 1:3])

# 对训练集和测试集的相应列应用转换
X_train[:, 1:3] = imputer.transform(X_train[:, 1:3])
X_test[:, 1:3] = imputer.transform(X_test[:, 1:3])

print("填充缺失值后的训练集 X_train:")
print(X_train)
print("填充缺失值后的测试集 X_test:")
print(X_test)