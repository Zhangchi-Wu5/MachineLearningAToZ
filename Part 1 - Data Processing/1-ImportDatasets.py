# 第1步：导入数据集
# 这个脚本展示了如何导入和检查数据集的基础操作

# 导入必要的库
import numpy as np           # 用于数值计算和数组操作
import matplotlib.pyplot as plt  # 用于绘制图表和数据可视化
import pandas as pd          # 用于数据处理和分析

# 导入数据集
# 使用pandas的read_csv函数读取CSV文件
dataset = pd.read_csv('datasets/Data.csv')

print("数据集基本信息:")
print(dataset)

# 创建两个重要的数据实体
X = dataset.iloc[:, :-1].values  # 特征矩阵：包含所有输入变量（除最后一列外）
y = dataset.iloc[:, -1].values   # 目标变量向量：我们要预测的变量（最后一列）

print("特征矩阵 X 的信息:")
print(X)
print("目标变量 y 的信息:")
print(y)