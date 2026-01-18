"""
预测脚本示例 - 演示如何加载保存的 Prophet 模型进行预测

使用方法:
  python predict.py

前置条件:
  1. 已运行 train_prophet.py 生成 models/prophet_model.json
  2. 已安装 prophet pandas
"""

import pandas as pd
from prophet.serialize import model_from_json
import sys
import os

print("=" * 60)
print("Prophet 模型加载与预测")
print("=" * 60)

# 检查模型文件是否存在
model_path = 'models/prophet_model.json'
if not os.path.exists(model_path):
    print(f"\n❌ 错误: 模型文件不存在: {model_path}")
    print("   请先运行 train_prophet.py 训练模型")
    sys.exit(1)

# 1. 从JSON加载模型（核心：model_from_json）
print(f"\n[1/4] 加载模型: {model_path}")
with open(model_path, 'r') as f:
    model = model_from_json(f.read())
print("✓ 模型加载成功")

# 2. 生成未来30天预测
print("\n[2/4] 生成未来30天预测...")
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
print(f"✓ 预测完成，生成 {len(forecast)} 条预测记录")

# 3. 获取未来预测值（不包含历史数据）
future_only = forecast.tail(30)
print("\n[3/4] 预测结果摘要:")
print("-" * 60)
print(f"预测日期范围: {future_only['ds'].min()} 至 {future_only['ds'].max()}")
print(f"平均日消耗: {future_only['yhat'].mean():.1f}g")
print(f"最大日消耗: {future_only['yhat'].max():.1f}g (日期: {future_only.loc[future_only['yhat'].idxmax(), 'ds']})")
print(f"最小日消耗: {future_only['yhat'].min():.1f}g (日期: {future_only.loc[future_only['yhat'].idxmin(), 'ds']})")

# 4. 库存剩余天数计算（业务逻辑）
print("\n[4/4] 库存剩余天数计算:")
print("-" * 60)

# 获取明天的预测值（最近一天）
tomorrow_prediction = future_only.iloc[0]
predicted_daily_avg = tomorrow_prediction['yhat']
predicted_lower = tomorrow_prediction['yhat_lower']
predicted_upper = tomorrow_prediction['yhat_upper']

print(f"明日预测消耗: {predicted_daily_avg:.1f}g")
print(f"  置信区间: [{predicted_lower:.1f}g, {predicted_upper:.1f}g]")

# 示例：不同库存水平的计算
print("\n库存剩余天数计算（基于平均预测值）:")
stock_levels = [3000, 5000, 7000, 10000]
for stock in stock_levels:
    remaining_days = stock / predicted_daily_avg
    status = ""
    if remaining_days < 7:
        status = "⚠️  警告：库存不足7天，建议立即补货！"
    elif remaining_days < 14:
        status = "⚠️  提醒：库存不足14天，请准备补货"
    else:
        status = "✓  库存充足"
    
    print(f"  库存 {stock:5d}g → 可支撑 {remaining_days:5.1f} 天  {status}")

# 使用保守估计（置信区间上限）
print("\n保守估计（使用置信区间上限）:")
for stock in stock_levels:
    remaining_days_conservative = stock / predicted_upper
    print(f"  库存 {stock:5d}g → 可支撑 {remaining_days_conservative:5.1f} 天")

# 保存详细预测结果
output_path = 'predictions/latest_forecast.csv'
future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(output_path, index=False)
print(f"\n✓ 详细预测已保存至: {output_path}")

print("\n" + "=" * 60)
print("预测完成！")
print("=" * 60)
print("\n提示：")
print("  - 预测值 yhat: 点预测")
print("  - yhat_lower/upper: 95%置信区间")
print("  - 建议使用置信区间上限做保守估计，避免缺货")
print()
