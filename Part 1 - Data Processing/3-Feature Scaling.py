# 第3步：特征缩放（完整数据预处理流程）
# 这个脚本展示了完整的数据预处理流程，包括特征缩放

# 导入必要的库
import numpy as np           # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import pandas as pd          # 用于数据处理和分析

# 导入数据集
dataset = pd.read_csv('datasets/Data.csv')
X = dataset.iloc[:, :-1].values  # 特征矩阵
y = dataset.iloc[:, -1].values   # 目标变量

# 处理缺失数据
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 数据集拆分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 对训练集和测试集应用缺失值填充
imputer.fit(X_train[:, 1:3])  # 在训练集上学习统计信息
X_train[:, 1:3] = imputer.transform(X_train[:, 1:3])  # 填充训练集缺失值
X_test[:, 1:3] = imputer.transform(X_test[:, 1:3])    # 填充测试集缺失值

# 编码分类数据 - 独立变量（特征）
# 使用One-Hot编码处理第0列的分类变量（如国家名称）
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train))  # 对训练集进行编码
X_test = np.array(ct.transform(X_test))        # 对测试集应用相同编码

# 编码分类数据 - 目标变量
# 使用标签编码处理二分类目标变量（如是/否）
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)  # 对训练集目标变量编码
y_test = le.transform(y_test)        # 对测试集目标变量应用相同编码

# 特征缩放
# 标准化处理：将特征缩放到相同的尺度，避免某些特征主导模型
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# 只对数值特征进行缩放（跳过One-Hot编码产生的0/1列）
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])  # 在训练集上拟合并转换
X_test[:, 3:] = sc.transform(X_test[:, 3:])        # 对测试集应用相同的缩放

print("特征缩放应用成功！")
print("训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)
print("数据预处理完成，可以开始训练模型！")