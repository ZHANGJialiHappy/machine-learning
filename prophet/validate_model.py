"""
模型交叉验证评估脚本 - 评估 Prophet 模型的预测性能

使用方法:
  python validate_model.py

说明:
  - 使用交叉验证评估模型在不同时间窗口的预测准确性
  - 计算 MAPE, RMSE, MAE 等指标
  - 评估未来 7天、14天、30天的预测性能
"""

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np

print("=" * 60)
print("Prophet 模型交叉验证评估")
print("=" * 60)

# 1. 加载训练数据
print("\n[1/4] 加载训练数据...")
df = pd.read_csv('data/train_data/daily_consumption.csv')
holidays = pd.read_csv('data/train_data/denmark_holidays_2025.csv')
print(f"✓ 数据加载完成: {len(df)} 条记录")

# 2. 训练模型
print("\n[2/4] 训练模型...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    holidays=holidays,
    seasonality_mode='additive',
    interval_width=0.95
)
model.fit(df)
print("✓ 模型训练完成")

# 3. 交叉验证
print("\n[3/4] 执行交叉验证（这可能需要1-2分钟）...")
print("  - initial: 180天用于训练")
print("  - period: 每7天滚动一次")
print("  - horizon: 预测未来30天")

try:
    df_cv = cross_validation(
        model, 
        initial='180 days',  # 初始训练数据量
        period='7 days',     # 每次向前移动7天
        horizon='30 days'    # 预测未来30天
    )
    print(f"✓ 交叉验证完成，生成 {len(df_cv)} 条验证记录")
    
    # 4. 计算性能指标
    print("\n[4/4] 计算性能指标...")
    df_metrics = performance_metrics(df_cv)
    
    print("\n" + "=" * 60)
    print("模型性能评估结果")
    print("=" * 60)
    
    # 按不同预测时间窗口聚合指标
    horizons_to_check = ['7 days', '14 days', '30 days']
    
    print("\n关键指标汇总:")
    print("-" * 60)
    print(f"{'时间窗口':<10} {'MAPE':<10} {'RMSE':<10} {'MAE':<10} {'Coverage':<10}")
    print("-" * 60)
    
    for horizon in horizons_to_check:
        # 筛选对应时间窗口的数据
        mask = df_metrics['horizon'] == pd.Timedelta(horizon)
        if mask.sum() > 0:
            subset = df_metrics[mask]
            mape = subset['mape'].mean()
            rmse = subset['rmse'].mean()
            mae = subset['mae'].mean()
            coverage = subset['coverage'].mean()
            
            print(f"{horizon:<10} {mape:<10.2%} {rmse:<10.1f} {mae:<10.1f} {coverage:<10.2%}")
    
    # 整体性能
    print("-" * 60)
    print(f"{'整体平均':<10} {df_metrics['mape'].mean():<10.2%} "
          f"{df_metrics['rmse'].mean():<10.1f} "
          f"{df_metrics['mae'].mean():<10.1f} "
          f"{df_metrics['coverage'].mean():<10.2%}")
    
    # 性能解读
    print("\n指标说明:")
    print("  - MAPE (Mean Absolute Percentage Error): 平均绝对百分比误差，越小越好")
    print("    * <10%: 优秀")
    print("    * 10-20%: 良好")
    print("    * 20-30%: 可接受")
    print("    * >30%: 需要改进")
    print("  - RMSE (Root Mean Square Error): 均方根误差，单位为克")
    print("  - MAE (Mean Absolute Error): 平均绝对误差，单位为克")
    print("  - Coverage: 置信区间覆盖率，理想值为95%")
    
    # 保存详细结果
    output_path = 'predictions/cross_validation_metrics.csv'
    df_metrics.to_csv(output_path, index=False)
    print(f"\n✓ 详细指标已保存至: {output_path}")
    
    # 性能评估建议
    avg_mape = df_metrics['mape'].mean()
    print("\n" + "=" * 60)
    print("模型评估结论:")
    print("=" * 60)
    
    if avg_mape < 0.10:
        print("✓ 模型性能优秀！预测准确度很高。")
    elif avg_mape < 0.20:
        print("✓ 模型性能良好，可用于生产环境。")
    elif avg_mape < 0.30:
        print("⚠️  模型性能可接受，但建议考虑：")
        print("   - 增加更多训练数据")
        print("   - 添加更多特征（如特殊事件、促销活动等）")
        print("   - 调整 seasonality_mode (additive vs multiplicative)")
    else:
        print("⚠️  模型性能需要改进，建议：")
        print("   - 检查数据质量（是否有异常值）")
        print("   - 增加训练数据量")
        print("   - 考虑添加更多外部特征")
        print("   - 调整模型参数或尝试其他模型")
    
    # 预测未来的不确定性提示
    print("\n预测不确定性:")
    horizon_7d = df_metrics[df_metrics['horizon'] == pd.Timedelta('7 days')]
    horizon_30d = df_metrics[df_metrics['horizon'] == pd.Timedelta('30 days')]
    
    if len(horizon_7d) > 0 and len(horizon_30d) > 0:
        mape_7d = horizon_7d['mape'].mean()
        mape_30d = horizon_30d['mape'].mean()
        increase = (mape_30d - mape_7d) / mape_7d * 100
        
        print(f"  - 7天预测 MAPE: {mape_7d:.2%}")
        print(f"  - 30天预测 MAPE: {mape_30d:.2%}")
        print(f"  - 误差增长: {increase:+.1f}%")
        print("  提示: 随着预测时间延长，不确定性会增加")

except Exception as e:
    print(f"\n❌ 交叉验证失败: {str(e)}")
    print("\n可能的原因:")
    print("  - 训练数据不足（需要至少 180 + 30 = 210天数据）")
    print("  - 解决方案: 减小 initial 参数，例如 '150 days'")

print("\n" + "=" * 60)
print("评估完成！")
print("=" * 60)
print()
