# 简单线性回归
# 这是最基础的回归算法，用于预测连续数值变量

# 导入必要的库
import numpy as np           # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import pandas as pd          # 用于数据处理

# 导入数据集
dataset = pd.read_csv('Salary_Data.csv')  # 工资数据集
X = dataset.iloc[:, :-1].values  # 特征：工作经验年数
y = dataset.iloc[:, -1].values   # 目标变量：工资

# 将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split
# test_size=1/3表示测试集占1/3，训练集占2/3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# 在训练集上训练简单线性回归模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # 创建线性回归对象
regressor.fit(X_train, y_train)  # 用训练数据拟合模型

# 对测试集进行预测
y_pred = regressor.predict(X_test)  # 预测测试集的工资

# 可视化训练集结果
plt.scatter(X_train, y_train, color = 'red')  # 散点图显示真实数据点
plt.plot(X_train, regressor.predict(X_train), color = 'blue')  # 线性回归线
plt.title('薪资 vs 工作经验 (训练集)')
plt.xlabel('工作经验年数')
plt.ylabel('薪资')
plt.show()

# 可视化测试集结果
plt.scatter(X_test, y_test, color = 'red')  # 测试集真实数据点
plt.plot(X_train, regressor.predict(X_train), color = 'blue')  # 同一条回归线
plt.title('薪资 vs 工作经验 (测试集)')
plt.xlabel('工作经验年数')
plt.ylabel('薪资')
plt.show()