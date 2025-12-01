"""
咖啡机使用数据(usage-extract.xlsx)统计分析
评估数据质量，为异常检测模型准备基础
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("咖啡机使用数据统计分析 (USAGE-EXTRACT.XLSX)")
print("="*80)

# ============================================================================
# 1. 数据加载
# ============================================================================
print("\n[1/6] 加载数据文件...")

try:
    usage_df = pd.read_excel('usage-extract.xlsx')
    print(f"✓ 成功加载: {len(usage_df):,} 条记录")
except Exception as e:
    print(f"✗ 数据加载失败: {e}")
    exit(1)

# 转换时间戳
usage_df['timestamp'] = pd.to_datetime(usage_df['timestamp'])

# ============================================================================
# 2. 基础数据质量统计
# ============================================================================
print("\n" + "="*80)
print("[2/6] 基础数据质量统计")
print("="*80)

print(f"\n列名: {list(usage_df.columns)}")
print(f"数据形状: {usage_df.shape[0]} 行 × {usage_df.shape[1]} 列")
print(f"\n唯一咖啡机数: {usage_df['external_id'].nunique()}")
print(f"时间跨度: {usage_df['timestamp'].min()} 至 {usage_df['timestamp'].max()}")
print(f"时长: {(usage_df['timestamp'].max() - usage_df['timestamp'].min()).days} 天")

# 数据概览
print(f"\n数据概览:")
print(usage_df.head(3).to_string())

# 缺失值检查
print("\n缺失值统计:")
missing = usage_df.isnull().sum()
missing_exists = False
for col in missing[missing > 0].index:
    pct = missing[col] / len(usage_df) * 100
    print(f"  ⚠ {col}: {missing[col]:,} ({pct:.2f}%)")
    missing_exists = True
if not missing_exists:
    print("  ✓ 无缺失值")

# 异常值检查
print("\n异常值检查:")
numeric_cols = ['water_volume_ml', 'coffee_g', 'cocoa_g', 'milk_powder_g']
for col in numeric_cols:
    neg_count = (usage_df[col] < 0).sum()
    zero_count = (usage_df[col] == 0).sum()
    if neg_count > 0:
        print(f"  ⚠ {col}: {neg_count} 条负值记录")
    print(f"  - {col}: {zero_count:,} 条零值记录 ({zero_count/len(usage_df)*100:.1f}%)")

# 基础统计描述
print(f"\n数值型字段统计描述:")
print(usage_df[numeric_cols].describe().to_string())

# ============================================================================
# 3. 咖啡机维度统计
# ============================================================================
print("\n" + "="*80)
print("[3/6] 咖啡机维度统计")
print("="*80)

# 每台机器的使用频率
machine_usage_counts = usage_df.groupby('external_id').size().sort_values(ascending=False)
print(f"\n每台机器使用频率统计:")
print(f"  总机器数: {len(machine_usage_counts)}")
print(f"  平均: {machine_usage_counts.mean():.1f} 杯")
print(f"  中位数: {machine_usage_counts.median():.1f} 杯")
print(f"  最小值: {machine_usage_counts.min()} 杯")
print(f"  最大值: {machine_usage_counts.max()} 杯")
print(f"  标准差: {machine_usage_counts.std():.1f}")

# Top 10 最活跃机器
print(f"\nTop 10 最活跃机器:")
for idx, (machine, count) in enumerate(machine_usage_counts.head(10).items(), 1):
    print(f"  {idx:2d}. {machine}: {count:,} 杯")

# 数据稀疏度分析
sparse_threshold = 50  # 少于50条记录认为数据不足
sparse_machines = machine_usage_counts[machine_usage_counts < sparse_threshold]
print(f"\n数据稀疏机器 (< {sparse_threshold} 条记录):")
print(f"  数量: {len(sparse_machines)} 台 ({len(sparse_machines)/len(machine_usage_counts)*100:.1f}%)")
if len(sparse_machines) > 0:
    print(f"  ⚠ 建议: 这些机器数据不足，可能不适合单独建模")

# 每台机器的活跃时间跨度
machine_time_span = usage_df.groupby('external_id')['timestamp'].agg(['min', 'max'])
machine_time_span['days'] = (machine_time_span['max'] - machine_time_span['min']).dt.days
print(f"\n每台机器活跃时间跨度:")
print(f"  平均: {machine_time_span['days'].mean():.1f} 天")
print(f"  中位数: {machine_time_span['days'].median():.1f} 天")
print(f"  最小值: {machine_time_span['days'].min()} 天")
print(f"  最大值: {machine_time_span['days'].max()} 天")

# 检查RETURNED机器
returned_machines = [m for m in usage_df['external_id'].unique() if 'RETURNED' in str(m)]
print(f"\nRETURNED测试机器: {len(returned_machines)} 台")
if len(returned_machines) > 0:
    print(f"  ⚠ 建议: 数据清洗时移除这些测试机器")
    print(f"  示例: {returned_machines[:5]}")

# 设备型号分布
print(f"\n设备型号分布:")
model_dist = usage_df['equipment_model_id'].value_counts()
for model, count in model_dist.items():
    machines_count = usage_df[usage_df['equipment_model_id'] == model]['external_id'].nunique()
    print(f"  型号 {model}: {count:,} 条记录 ({machines_count} 台机器)")

# ============================================================================
# 4. 时间分布统计
# ============================================================================
print("\n" + "="*80)
print("[4/6] 时间分布统计")
print("="*80)

# 提取时间特征
usage_df['hour'] = usage_df['timestamp'].dt.hour
usage_df['day_of_week'] = usage_df['timestamp'].dt.dayofweek
usage_df['date'] = usage_df['timestamp'].dt.date

# 每小时使用分布
hourly_dist = usage_df['hour'].value_counts().sort_index()
print("\n每小时使用分布 (Top 10 高峰时段):")
for hour, count in hourly_dist.sort_values(ascending=False).head(10).items():
    print(f"  {hour:02d}:00 - {count:,} 杯 ({count/len(usage_df)*100:.2f}%)")

# 每天使用分布
daily_dist = usage_df.groupby('date').size()
print(f"\n每日使用量统计:")
print(f"  平均: {daily_dist.mean():.1f} 杯/天")
print(f"  中位数: {daily_dist.median():.1f} 杯/天")
print(f"  最小值: {daily_dist.min()} 杯/天")
print(f"  最大值: {daily_dist.max()} 杯/天")
print(f"  标准差: {daily_dist.std():.1f}")

# 工作日 vs 周末
weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
print("\n周几使用分布:")
dow_dist = usage_df['day_of_week'].value_counts().sort_index()
for dow, count in dow_dist.items():
    print(f"  {weekday_names[dow]}: {count:,} 杯 ({count/len(usage_df)*100:.1f}%)")

# 使用间隔统计（按机器分组）
print("\n使用间隔统计（相邻两杯之间，取样本分析）:")
intervals = []
# 随机抽样一些机器来计算间隔，避免处理太久
sample_machines = usage_df['external_id'].unique()[:50]
for machine in sample_machines:
    machine_data = usage_df[usage_df['external_id'] == machine].sort_values('timestamp')
    if len(machine_data) > 1:
        time_diffs = machine_data['timestamp'].diff().dt.total_seconds() / 60  # 转换为分钟
        intervals.extend(time_diffs.dropna().tolist())

if len(intervals) > 0:
    intervals = pd.Series(intervals)
    print(f"  平均间隔: {intervals.mean():.1f} 分钟")
    print(f"  中位数: {intervals.median():.1f} 分钟")
    print(f"  25%分位: {intervals.quantile(0.25):.1f} 分钟")
    print(f"  75%分位: {intervals.quantile(0.75):.1f} 分钟")
    
    # 异常短间隔（可能是连续制作）
    short_intervals = intervals[intervals < 1]  # 小于1分钟
    print(f"  异常短间隔 (< 1分钟): {len(short_intervals):,} 次 ({len(short_intervals)/len(intervals)*100:.2f}%)")

# ============================================================================
# 5. 使用量分析（水量、咖啡量等）
# ============================================================================
print("\n" + "="*80)
print("[5/6] 使用量分析")
print("="*80)

print("\n【水量统计 (water_volume_ml)】")
water_stats = usage_df['water_volume_ml'].describe()
print(f"  平均值: {water_stats['mean']:.1f} ml")
print(f"  中位数: {water_stats['50%']:.1f} ml")
print(f"  标准差: {water_stats['std']:.1f} ml")
print(f"  最小值: {water_stats['min']:.1f} ml")
print(f"  最大值: {water_stats['max']:.1f} ml")
print(f"  25%分位: {water_stats['25%']:.1f} ml")
print(f"  75%分位: {water_stats['75%']:.1f} ml")

print("\n【咖啡用量统计 (coffee_g)】")
coffee_stats = usage_df['coffee_g'].describe()
print(f"  平均值: {coffee_stats['mean']:.2f} g")
print(f"  中位数: {coffee_stats['50%']:.2f} g")
print(f"  标准差: {coffee_stats['std']:.2f} g")
print(f"  最小值: {coffee_stats['min']:.2f} g")
print(f"  最大值: {coffee_stats['max']:.2f} g")
print(f"  零值比例: {(usage_df['coffee_g'] == 0).sum() / len(usage_df) * 100:.1f}%")

print("\n【可可用量统计 (cocoa_g)】")
cocoa_nonzero = usage_df[usage_df['cocoa_g'] > 0]
print(f"  非零记录: {len(cocoa_nonzero):,} ({len(cocoa_nonzero)/len(usage_df)*100:.1f}%)")
if len(cocoa_nonzero) > 0:
    print(f"  非零均值: {cocoa_nonzero['cocoa_g'].mean():.2f} g")
    print(f"  非零标准差: {cocoa_nonzero['cocoa_g'].std():.2f} g")

print("\n【奶粉用量统计 (milk_powder_g)】")
milk_nonzero = usage_df[usage_df['milk_powder_g'] > 0]
print(f"  非零记录: {len(milk_nonzero):,} ({len(milk_nonzero)/len(usage_df)*100:.1f}%)")
if len(milk_nonzero) > 0:
    print(f"  非零均值: {milk_nonzero['milk_powder_g'].mean():.2f} g")
    print(f"  非零标准差: {milk_nonzero['milk_powder_g'].std():.2f} g")

# 按咖啡机分组的使用量统计
print("\n【按机器分组的水量变异系数 (CV)】")
machine_water_stats = usage_df.groupby('external_id')['water_volume_ml'].agg(['mean', 'std', 'count'])
machine_water_stats = machine_water_stats[machine_water_stats['count'] >= 10]  # 至少10条记录
machine_water_stats['cv'] = machine_water_stats['std'] / machine_water_stats['mean']
print(f"  分析机器数: {len(machine_water_stats)} (至少10条记录)")
print(f"  平均CV: {machine_water_stats['cv'].mean():.3f}")
print(f"  中位CV: {machine_water_stats['cv'].median():.3f}")
high_cv_machines = machine_water_stats[machine_water_stats['cv'] > 0.5]
print(f"  CV > 0.5 的机器: {len(high_cv_machines)} 台 ({len(high_cv_machines)/len(machine_water_stats)*100:.1f}%)")
if len(high_cv_machines) > 0:
    print(f"  ⚠ 这些机器的水量使用不稳定，可能存在异常")

# ============================================================================
# 6. 饮品类型统计
# ============================================================================
print("\n" + "="*80)
print("[6/6] 饮品类型统计")
print("="*80)

drink_counts = usage_df['name'].value_counts()
print(f"\n饮品种类总数: {len(drink_counts)}")
print(f"\nTop 20 热门饮品:")
for idx, (drink, count) in enumerate(drink_counts.head(20).items(), 1):
    pct = count / len(usage_df) * 100
    print(f"  {idx:2d}. {drink:35s}: {count:6,} 杯 ({pct:5.2f}%)")

# 按饮品类型的平均用量
print(f"\n主要饮品的平均原料用量 (Top 8):")
for drink in drink_counts.head(8).index:
    drink_data = usage_df[usage_df['name'] == drink]
    print(f"\n  【{drink}】 ({len(drink_data):,} 杯)")
    print(f"    水量: {drink_data['water_volume_ml'].mean():6.1f} ± {drink_data['water_volume_ml'].std():5.1f} ml (CV: {drink_data['water_volume_ml'].std()/drink_data['water_volume_ml'].mean():.3f})")
    print(f"    咖啡: {drink_data['coffee_g'].mean():6.2f} ± {drink_data['coffee_g'].std():5.2f} g  (CV: {drink_data['coffee_g'].std()/drink_data['coffee_g'].mean() if drink_data['coffee_g'].mean() > 0 else 0:.3f})")
    if drink_data['milk_powder_g'].sum() > 0:
        milk_data = drink_data[drink_data['milk_powder_g'] > 0]
        print(f"    奶粉: {milk_data['milk_powder_g'].mean():6.2f} ± {milk_data['milk_powder_g'].std():5.2f} g  ({len(milk_data)} 杯使用)")
    if drink_data['cocoa_g'].sum() > 0:
        cocoa_data = drink_data[drink_data['cocoa_g'] > 0]
        print(f"    可可: {cocoa_data['cocoa_g'].mean():6.2f} ± {cocoa_data['cocoa_g'].std():5.2f} g  ({len(cocoa_data)} 杯使用)")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("数据质量评估总结")
print("="*80)

print("\n【✓ 优点】")
print(f"1. 数据量充足: {len(usage_df):,} 条记录")
print(f"2. 机器覆盖广: {usage_df['external_id'].nunique()} 台机器")
print(f"3. 时间跨度合理: {(usage_df['timestamp'].max() - usage_df['timestamp'].min()).days} 天")
print(f"4. 特征丰富: 水量、咖啡量、可可、奶粉、饮品类型")
print(f"5. 饮品多样: {len(drink_counts)} 种不同饮品")

print("\n【⚠ 需要注意】")
issues = []
if len(sparse_machines) > 0:
    issues.append(f"1. {len(sparse_machines)} 台机器数据稀疏 (< {sparse_threshold} 条)")
if len(returned_machines) > 0:
    issues.append(f"2. {len(returned_machines)} 台RETURNED测试机器需要清洗")
if len(high_cv_machines) > 0:
    issues.append(f"3. {len(high_cv_machines)} 台机器用量不稳定 (CV > 0.5)")

if issues:
    for issue in issues:
        print(f"{issue}")
else:
    print("数据质量良好，无明显问题")

print("\n【适合的异常检测应用】")
print("✓ 水量异常检测 - 检测异常的水量消耗")
print("✓ 咖啡用量异常 - 按饮品类型检测异常用量")
print("✓ 使用频率异常 - 检测突然的使用频率变化")
print("✓ 配方一致性 - 检测同类饮品的原料配比异常")
print("✓ 时间模式异常 - 检测异常的使用时间模式")

print("\n【推荐的下一步】")
print("1. 数据清洗: 移除RETURNED机器，过滤稀疏数据")
print("2. 特征工程: 构建时间窗口特征、统计特征")
print("3. 模型选择: Isolation Forest 或 Autoencoder")
print("4. 异常阈值设定: 基于统计结果设置合理阈值")

# ============================================================================
# 保存统计结果到CSV
# ============================================================================
print("\n" + "="*80)
print("保存统计结果到CSV文件...")
print("="*80)

# 1. 机器级别统计
machine_stats_df = pd.DataFrame({
    'external_id': machine_usage_counts.index,
    'total_cups': machine_usage_counts.values,
    'first_use': machine_time_span['min'].values,
    'last_use': machine_time_span['max'].values,
    'active_days': machine_time_span['days'].values,
})

# 添加平均用量
machine_avg_water = usage_df.groupby('external_id')['water_volume_ml'].mean()
machine_avg_coffee = usage_df.groupby('external_id')['coffee_g'].mean()
machine_stats_df['avg_water_ml'] = machine_stats_df['external_id'].map(machine_avg_water)
machine_stats_df['avg_coffee_g'] = machine_stats_df['external_id'].map(machine_avg_coffee)

# 添加变异系数
machine_stats_df['water_cv'] = machine_stats_df['external_id'].map(
    machine_water_stats['cv'] if len(machine_water_stats) > 0 else {}
)

machine_stats_df.to_csv('machine_statistics.csv', index=False)
print(f"✓ machine_statistics.csv - {len(machine_stats_df)} 台机器的统计数据")

# 2. 饮品类型统计
drink_stats_list = []
for drink in drink_counts.index:
    drink_data = usage_df[usage_df['name'] == drink]
    drink_stats_list.append({
        'drink_name': drink,
        'total_cups': len(drink_data),
        'percentage': len(drink_data) / len(usage_df) * 100,
        'avg_water_ml': drink_data['water_volume_ml'].mean(),
        'std_water_ml': drink_data['water_volume_ml'].std(),
        'avg_coffee_g': drink_data['coffee_g'].mean(),
        'std_coffee_g': drink_data['coffee_g'].std(),
        'avg_milk_powder_g': drink_data['milk_powder_g'].mean(),
        'avg_cocoa_g': drink_data['cocoa_g'].mean(),
    })

drink_stats_df = pd.DataFrame(drink_stats_list)
drink_stats_df.to_csv('drink_statistics.csv', index=False)
print(f"✓ drink_statistics.csv - {len(drink_stats_df)} 种饮品的统计数据")

# 3. 每日使用统计
daily_stats_df = usage_df.groupby('date').agg({
    'external_id': 'count',  # 总杯数
    'water_volume_ml': 'sum',
    'coffee_g': 'sum',
    'milk_powder_g': 'sum',
    'cocoa_g': 'sum'
}).reset_index()
daily_stats_df.columns = ['date', 'total_cups', 'total_water_ml', 'total_coffee_g', 'total_milk_powder_g', 'total_cocoa_g']
daily_stats_df.to_csv('daily_usage.csv', index=False)
print(f"✓ daily_usage.csv - {len(daily_stats_df)} 天的使用统计")

# 4. 每小时分布统计
hourly_stats_df = pd.DataFrame({
    'hour': hourly_dist.index,
    'cup_count': hourly_dist.values,
    'percentage': hourly_dist.values / len(usage_df) * 100
})
hourly_stats_df.to_csv('hourly_distribution.csv', index=False)
print(f"✓ hourly_distribution.csv - 24小时使用分布")

# 5. 总体摘要统计
summary_stats = {
    'metric': [
        'total_records', 'unique_machines', 'date_range_days',
        'avg_water_ml', 'avg_coffee_g', 'avg_milk_powder_g', 'avg_cocoa_g',
        'total_drink_types', 'sparse_machines_count', 'returned_machines_count',
        'high_cv_machines_count'
    ],
    'value': [
        len(usage_df),
        usage_df['external_id'].nunique(),
        (usage_df['timestamp'].max() - usage_df['timestamp'].min()).days,
        usage_df['water_volume_ml'].mean(),
        usage_df['coffee_g'].mean(),
        usage_df['milk_powder_g'].mean(),
        usage_df['cocoa_g'].mean(),
        len(drink_counts),
        len(sparse_machines),
        len(returned_machines),
        len(high_cv_machines) if len(high_cv_machines) > 0 else 0
    ]
}
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('summary_statistics.csv', index=False)
print(f"✓ summary_statistics.csv - 总体摘要统计")

# 6. 异常机器列表
if len(sparse_machines) > 0 or len(returned_machines) > 0 or len(high_cv_machines) > 0:
    anomaly_machines = []
    
    for machine in sparse_machines.index:
        anomaly_machines.append({
            'external_id': machine,
            'issue_type': 'sparse_data',
            'cups': sparse_machines[machine],
            'description': f'数据稀疏 (< {sparse_threshold} 条)'
        })
    
    for machine in returned_machines:
        anomaly_machines.append({
            'external_id': machine,
            'issue_type': 'returned',
            'cups': machine_usage_counts.get(machine, 0),
            'description': '测试/退货机器'
        })
    
    if len(high_cv_machines) > 0:
        for machine in high_cv_machines.index:
            anomaly_machines.append({
                'external_id': machine,
                'issue_type': 'high_cv',
                'cups': machine_usage_counts.get(machine, 0),
                'description': f'用量不稳定 (CV={high_cv_machines.loc[machine, "cv"]:.3f})'
            })
    
    anomaly_df = pd.DataFrame(anomaly_machines)
    anomaly_df.to_csv('anomaly_machines.csv', index=False)
    print(f"✓ anomaly_machines.csv - {len(anomaly_df)} 条异常机器记录")

print("\n所有统计结果已保存到CSV文件！")

print("\n" + "="*80)
print("统计分析完成！")
print("="*80)
