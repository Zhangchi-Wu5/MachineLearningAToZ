# 第2步：数据集拆分
# 这个脚本展示了如何将数据集分为训练集和测试集

# 导入必要的库
import numpy as np           # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import pandas as pd          # 用于数据处理和分析

# 导入数据集
dataset = pd.read_csv('datasets/Data.csv')

print("数据集信息:")
print(dataset)

# 提取特征矩阵和目标变量
X = dataset.iloc[:, :-1].values  # 特征矩阵（输入变量）
y = dataset.iloc[:, -1].values   # 目标变量（我们要预测的值）

# 数据集拆分为训练集和测试集
# 这是机器学习中的重要步骤，用于评估模型性能
from sklearn.model_selection import train_test_split

# 参数说明：
# X, y: 要拆分的特征矩阵和目标变量
# test_size=0.2: 测试集占总数据的20%，训练集占80%
# random_state=1: 设置随机种子，确保每次运行结果一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("数据拆分完成！")
print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}") 