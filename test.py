import joblib
import numpy as np

# 1. 加载模型
clf = joblib.load('models/random_forest_model.pkl')

# 2. 准备一个测试样本（鸢尾花的4个特征）
# 格式: [花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度]
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # 一个典型的山鸢尾

# 3. 预测
prediction = clf.predict(sample)
print(f"预测类别: {prediction[0]}")

# 4. 查看预测概率
probabilities = clf.predict_proba(sample)
print(f"各类别概率: {probabilities[0]}")
print(f"类别0概率: {probabilities[0][0]:.2%}")
print(f"类别1概率: {probabilities[0][1]:.2%}")
print(f"类别2概率: {probabilities[0][2]:.2%}")