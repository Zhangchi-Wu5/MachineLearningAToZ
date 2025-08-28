# 数据预处理工具 - 完整模板
# 这是机器学习项目中数据预处理的标准模板

# 导入必要的库
import numpy as np           # 用于数值计算和数组操作
import matplotlib.pyplot as plt  # 用于数据可视化
import pandas as pd          # 用于数据处理和分析

# 导入数据集
dataset = pd.read_csv('Data.csv')  # 读取CSV文件
X = dataset.iloc[:, :-1].values    # 特征矩阵：所有行，除最后一列外的所有列
y = dataset.iloc[:, -1].values     # 目标变量：所有行，最后一列
print("特征矩阵 X:")
print(X)
print("目标变量 y:")
print(y)

# 处理缺失数据
# 使用SimpleImputer类来替换数据集中的缺失值
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # 用平均值填充NaN
imputer.fit(X[:, 1:3])  # 对第1列和第2列（年龄和薪资）计算统计信息
X[:, 1:3] = imputer.transform(X[:, 1:3])  # 应用转换，填充缺失值
print("填充缺失值后的特征矩阵:")
print(X)

# 编码分类数据
# 对独立变量（特征）进行编码
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# 创建列转换器：对第0列进行One-Hot编码，其余列保持不变
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))  # 应用编码并转换为NumPy数组
print("独立变量编码后的特征矩阵:")
print(X)

# 对目标变量进行编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()  # 创建标签编码器
y = le.fit_transform(y)  # 将分类标签转换为数值
print("编码后的目标变量:")
print(y)

# 将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split
# test_size=0.2：测试集占20%，random_state=1：设置随机种子保证结果可重现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print("训练集特征:")
print(X_train)
print("测试集特征:")
print(X_test)
print("训练集标签:")
print(y_train)
print("测试集标签:")
print(y_test)

# 特征缩放
# 使用标准化将特征缩放到相同的尺度范围
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  # 创建标准化器
# 只对连续数值特征进行缩放（跳过One-Hot编码的虚拟变量列）
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])  # 拟合并转换训练集
X_test[:, 3:] = sc.transform(X_test[:, 3:])        # 只转换测试集（不重新拟合）
print("标准化后的训练集特征:")
print(X_train)
print("标准化后的测试集特征:")
print(X_test)