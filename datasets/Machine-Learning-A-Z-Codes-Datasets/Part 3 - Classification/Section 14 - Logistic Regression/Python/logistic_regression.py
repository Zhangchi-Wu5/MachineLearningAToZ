# 逻辑回归
# 用于二分类问题的线性分类算法，输出概率值

# 导入必要的库
import numpy as np           # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import pandas as pd          # 用于数据处理

# 导入数据集
dataset = pd.read_csv('Social_Network_Ads.csv')  # 社交网络广告数据集
X = dataset.iloc[:, :-1].values  # 特征：年龄和预估薪资
y = dataset.iloc[:, -1].values   # 目标变量：是否购买产品（0或1）

# 将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print("训练集特征：")
print(X_train)
print("训练集标签：")
print(y_train)
print("测试集特征：")
print(X_test)
print("测试集标签：")
print(y_test)

# 特征缩放
# 逻辑回归对特征尺度敏感，需要标准化处理
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # 拟合并转换训练集
X_test = sc.transform(X_test)        # 转换测试集（使用训练集的缩放参数）
print("标准化后的训练集：")
print(X_train)
print("标准化后的测试集：")
print(X_test)

# 在训练集上训练逻辑回归模型
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)  # 创建逻辑回归分类器
classifier.fit(X_train, y_train)  # 训练模型

# 预测新的结果
# 例如：预测30岁、年薪87000的用户是否会购买产品
print("单个预测结果：")
print(classifier.predict(sc.transform([[30,87000]])))

# 预测测试集结果
y_pred = classifier.predict(X_test)  # 对测试集进行预测
# 并排显示预测值和真实值进行比较
print("预测值 vs 真实值：")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# 创建混淆矩阵
# 用于评估分类模型的性能
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)  # 生成混淆矩阵
print("混淆矩阵：")
print(cm)
print("准确率：", accuracy_score(y_test, y_pred))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()