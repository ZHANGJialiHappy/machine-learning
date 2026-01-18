import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import os

print("=" * 60)
print("Prophet 咖啡豆消耗预测模型训练")
print("=" * 60)

# 1. 加载训练数据
print("\n[1/8] 加载训练数据...")
df = pd.read_csv('data/train_data/daily_consumption.csv')
print(f"✓ 训练数据加载完成: {len(df)} 条记录")
print(f"  日期范围: {df['ds'].min()} 至 {df['ds'].max()}")
print(f"  平均消耗: {df['y'].mean():.1f}g/天")
print(f"  最大消耗: {df['y'].max():.0f}g (日期: {df.loc[df['y'].idxmax(), 'ds']})")
print(f"  最小消耗: {df['y'].min():.0f}g (日期: {df.loc[df['y'].idxmin(), 'ds']})")

# 2. 加载节假日数据
print("\n[2/8] 加载节假日数据...")
holidays = pd.read_csv('data/train_data/denmark_holidays_2025.csv')
print(f"✓ 节假日数据加载完成: {len(holidays)} 个节假日")
print(f"  节假日列表: {', '.join(holidays['holiday'].tolist()[:5])}...")

# 3. 配置并训练模型
print("\n[3/8] 配置 Prophet 模型...")
model = Prophet(
    yearly_seasonality=True,      # 年度季节性
    weekly_seasonality=True,      # 工作日/周末差异
    daily_seasonality=False,      # 日数据不需要
    holidays=holidays,            # 丹麦节假日
    seasonality_mode='additive',  # 加法模式
    interval_width=0.95           # 95%置信区间
)
print("✓ 模型配置完成")
print("  - yearly_seasonality: True (捕捉年度季节性)")
print("  - weekly_seasonality: True (工作日/周末差异)")
print("  - holidays: 丹麦节假日 (13个)")
print("  - seasonality_mode: additive")

print("\n[4/8] 训练模型（这可能需要几秒钟）...")
model.fit(df)
print("✓ 模型训练完成！")

# 4. 创建输出目录
print("\n[5/8] 创建输出目录...")
os.makedirs('models', exist_ok=True)
os.makedirs('predictions', exist_ok=True)
print("✓ 目录创建完成: models/, predictions/")

# 5. 序列化保存模型（核心：model_to_json）
print("\n[6/8] 序列化并保存模型...")
with open('models/prophet_model.json', 'w') as f:
    f.write(model_to_json(model))
model_size = os.path.getsize('models/prophet_model.json') / 1024  # KB
print(f"✓ 模型已保存至: models/prophet_model.json")
print(f"  文件大小: {model_size:.1f} KB")

# 6. 生成未来30天预测（作为示例输出）
print("\n[7/8] 生成未来30天预测...")
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# 只取未来30天的预测结果
future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)

# 7. 保存预测结果
future_forecast.to_csv('predictions/forecast_30days.csv', index=False)
print("✓ 预测结果已保存至: predictions/forecast_30days.csv")

# 8. 输出关键信息
print("\n[8/8] 预测结果摘要:")
print("-" * 60)
future_7days = forecast.tail(7)['yhat']
future_30days = forecast.tail(30)['yhat']
print(f"未来 7天预测平均消耗:  {future_7days.mean():.1f}g/天")
print(f"未来30天预测平均消耗:  {future_30days.mean():.1f}g/天")
print(f"未来30天预测最大消耗:  {future_30days.max():.1f}g/天")
print(f"未来30天预测最小消耗:  {future_30days.min():.1f}g/天")

# 示例：库存计算
print("\n" + "=" * 60)
print("示例：库存剩余天数计算")
print("=" * 60)
example_stock = 5000
predicted_avg = future_30days.mean()
remaining_days = example_stock / predicted_avg
print(f"假设当前库存: {example_stock}g")
print(f"预测日均消耗: {predicted_avg:.1f}g")
print(f"预计可支撑天数: {remaining_days:.1f}天")

if remaining_days < 7:
    print("⚠️  警告：库存不足7天，建议立即补货！")
elif remaining_days < 14:
    print("⚠️  提醒：库存不足14天，请准备补货")
else:
    print("✓  库存充足")

print("\n" + "=" * 60)
print("训练完成！")
print("=" * 60)
print("\n下一步：")
print("  1. 查看预测结果: predictions/forecast_30days.csv")
print("  2. 使用模型预测: 用 model_from_json() 加载 models/prophet_model.json")
print("  3. 定期更新模型: 追加新数据到 daily_consumption.csv 后重新训练")
print()
