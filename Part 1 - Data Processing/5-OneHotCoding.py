# 独热编码（One-Hot Encoding）实践
# 学习目标：理解如何处理分类变量，避免虚拟变量陷阱

# 导入必要的库
import pandas as pd              # 用于数据处理和分析
import numpy as np               # 用于数值计算
from sklearn.compose import ColumnTransformer      # 用于列转换
from sklearn.preprocessing import OneHotEncoder     # 用于独热编码

# 导入数据集
print("=== 步骤1：加载数据集 ===")
dataset = pd.read_csv('datasets/Data.csv')  # 读取CSV文件
print("原始数据集：")
print(dataset)
print(f"数据集形状：{dataset.shape}")

# 分离特征和目标变量
print("\n=== 步骤2：分离特征和目标变量 ===")
X = dataset.iloc[:, :-1].values  # 特征矩阵：所有行，除最后一列外的所有列
y = dataset.iloc[:, -1].values   # 目标变量：所有行，最后一列
print("特征矩阵 X（编码前）：")
print(X)
print("目标变量 y：")
print(y)

# 对分类特征进行独热编码
print("\n=== 步骤3：应用独热编码 ===")
# 创建列转换器：对第0列（Country）进行独热编码，其余列保持不变
# transformers: 指定转换器列表
# 'encoder': 转换器名称
# OneHotEncoder(): 独热编码器
# [0]: 要编码的列索引（Country列）
# remainder='passthrough': 其余列保持原样传递
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# 应用转换并转换为NumPy数组
X = np.array(ct.fit_transform(X))
print("独热编码后的特征矩阵 X：")
print(X)
print(f"编码后形状：{X.shape}")

# 显示编码结果解释
print("\n=== 编码结果解释 ===")
print("原始Country列的唯一值：", dataset['Country'].unique())
print("编码后前3列代表：")
print("- 第1列：France (1表示是，0表示否)")
print("- 第2列：Germany (1表示是，0表示否)")  
print("- 第3列：Spain (1表示是，0表示否)")
print("- 第4列：Age（保持原样）")
print("- 第5列：Salary（保持原样）")

print("\n=== 独热编码完成 ===")
print("特征数量从3个增加到5个")
print("成功避免了虚拟变量陷阱（不需要手动删除一列）")