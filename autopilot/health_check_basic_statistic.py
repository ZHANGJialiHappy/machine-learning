"""
Model 1 咖啡机数据质量统计分析
评估每台机器的数据是否适合进行异常检测模型训练
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Model 1 咖啡机数据质量统计分析")
print("=" * 80)

# ============================================================================
# 1. 加载三个数据文件并转换时间字段
# ============================================================================
print("\n[1/8] 加载数据文件...")

try:
    # 加载 usage 数据
    usage_df = pd.read_excel('data/model1/usage-extract.xlsx')
    usage_df['timestamp'] = pd.to_datetime(usage_df['timestamp'])
    usage_df['date'] = usage_df['timestamp'].dt.date
    print(f"✓ usage-extract.xlsx: {len(usage_df):,} 条记录")
    
    # 加载 error 数据
    error_df = pd.read_csv('data/model1/error-extract.csv')
    error_df['timestamp'] = pd.to_datetime(error_df['timestamp'])
    error_df['date'] = error_df['timestamp'].dt.date
    print(f"✓ error-extract.csv: {len(error_df):,} 条记录")
    
    # 加载 cleaning 数据
    cleaning_df = pd.read_csv('data/model1/cleaning-extract.csv')
    cleaning_df['timestamp'] = pd.to_datetime(cleaning_df['timestamp'])
    cleaning_df['date'] = cleaning_df['timestamp'].dt.date
    print(f"✓ cleaning-extract.csv: {len(cleaning_df):,} 条记录")
    
except Exception as e:
    print(f"✗ 数据加载失败: {e}")
    exit(1)

print(f"\n唯一咖啡机数: {usage_df['external_id'].nunique()}")
print(f"数据时间跨度: {usage_df['timestamp'].min()} 至 {usage_df['timestamp'].max()}")

# ============================================================================
# 2. 按 external_id + date 聚合每日出杯数
# ============================================================================
print("\n[2/8] 按机器和日期聚合每日出杯数...")

# 每台机器每天的出杯数
daily_cups = usage_df.groupby(['external_id', 'date']).size().reset_index(name='cups_cnt')
print(f"✓ 生成 {len(daily_cups):,} 条每日统计记录")
print(f"  示例: {daily_cups.head(3).to_dict('records')}")

# ============================================================================
# 3. 计算每台机器的活跃度统计指标
# ============================================================================
print("\n[3/8] 计算每台机器的活跃度统计指标...")

# 每台机器的活跃度
activity_stats = []
for machine_id in usage_df['external_id'].unique():
    machine_data = daily_cups[daily_cups['external_id'] == machine_id]
    
    # 活跃天数（有出杯记录的天数）
    active_days = len(machine_data)
    
    # 总天数（从第一条到最后一条记录的日历天数）
    first_date = machine_data['date'].min()
    last_date = machine_data['date'].max()
    total_days = (pd.to_datetime(last_date) - pd.to_datetime(first_date)).days + 1
    
    # 活跃度比例
    active_days_ratio = active_days / total_days if total_days > 0 else 0
    
    activity_stats.append({
        'external_id': machine_id,
        'active_days': active_days,
        'total_days': total_days,
        'active_days_ratio': active_days_ratio,
        'first_date': first_date,
        'last_date': last_date
    })

activity_df = pd.DataFrame(activity_stats)
print(f"✓ 计算 {len(activity_df)} 台机器的活跃度统计")

# ============================================================================
# 4. 计算每台机器的出杯统计指标
# ============================================================================
print("\n[4/8] 计算每台机器的出杯统计指标...")

cups_stats = daily_cups.groupby('external_id')['cups_cnt'].agg([
    ('mean_cups_cnt', 'mean'),
    ('std_cups_cnt', 'std'),
    ('min_cups_cnt', 'min'),
    ('max_cups_cnt', 'max'),
    ('total_cups', 'sum')
]).reset_index()

# 计算变异系数 (CV)
cups_stats['cv_cups'] = cups_stats['std_cups_cnt'] / cups_stats['mean_cups_cnt']
# 处理 std = 0 的情况
cups_stats['std_cups_cnt'] = cups_stats['std_cups_cnt'].fillna(0)
cups_stats['cv_cups'] = cups_stats['cv_cups'].fillna(0)

print(f"✓ 计算 {len(cups_stats)} 台机器的出杯统计")

# ============================================================================
# 5. 统计每台机器的错误和清洗数据
# ============================================================================
print("\n[5/8] 统计每台机器的错误和清洗数据...")

# 错误统计
error_stats = error_df.groupby('external_id').agg(
    total_errors=('error_code', 'count'),
    error_days=('date', 'nunique'),
    fatal_error_cnt=('fatal_error', lambda x: x.sum())
).reset_index()

# 清洗统计
cleaning_stats = cleaning_df.groupby('external_id').agg(
    total_cleanings=('cleaning_code', 'count'),
    cleaning_days=('date', 'nunique')
).reset_index()

print(f"✓ 错误统计: {len(error_stats)} 台机器有错误记录")
print(f"✓ 清洗统计: {len(cleaning_stats)} 台机器有清洗记录")

# ============================================================================
# 6. 合并所有统计指标到一个 DataFrame
# ============================================================================
print("\n[6/8] 合并所有统计指标...")

# 从活跃度统计开始
machine_stats = activity_df.copy()

# 合并出杯统计
machine_stats = machine_stats.merge(cups_stats, on='external_id', how='left')

# 合并错误统计
machine_stats = machine_stats.merge(error_stats, on='external_id', how='left')

# 合并清洗统计
machine_stats = machine_stats.merge(cleaning_stats, on='external_id', how='left')

# 填充缺失值（没有错误或清洗记录的机器）
machine_stats['total_errors'] = machine_stats['total_errors'].fillna(0).astype(int)
machine_stats['error_days'] = machine_stats['error_days'].fillna(0).astype(int)
machine_stats['fatal_error_cnt'] = machine_stats['fatal_error_cnt'].fillna(0).astype(int)
machine_stats['total_cleanings'] = machine_stats['total_cleanings'].fillna(0).astype(int)
machine_stats['cleaning_days'] = machine_stats['cleaning_days'].fillna(0).astype(int)

print(f"✓ 合并完成，共 {len(machine_stats)} 台机器")

# ============================================================================
# 7. 应用规则判断每台机器是否适合进模型
# ============================================================================
print("\n[7/8] 应用规则判断每台机器是否适合进模型...")

# 判断规则
machine_stats['is_suitable'] = (
    (machine_stats['active_days_ratio'] >= 0.5) &  # 活跃度 >= 50%
    (machine_stats['mean_cups_cnt'] >= 5) &        # 日均出杯 >= 5
    (machine_stats['std_cups_cnt'] > 0) &          # 有变化（非僵尸机器）
    (machine_stats['cv_cups'] < 3)                 # 不是极度不稳定
)

# 标记问题类型
def get_issues(row):
    issues = []
    if row['active_days_ratio'] < 0.5:
        issues.append('低活跃度')
    if row['mean_cups_cnt'] < 5:
        issues.append('出杯不足')
    if row['std_cups_cnt'] == 0:
        issues.append('无变化')
    if row['cv_cups'] >= 3:
        issues.append('极度不稳定')
    return '; '.join(issues) if issues else '正常'

machine_stats['issues'] = machine_stats.apply(get_issues, axis=1)

suitable_count = machine_stats['is_suitable'].sum()
unsuitable_count = len(machine_stats) - suitable_count

print(f"✓ ✅ 适合进模型: {suitable_count} 台 ({suitable_count/len(machine_stats)*100:.1f}%)")
print(f"✓ ⚠️ 不适合进模型: {unsuitable_count} 台 ({unsuitable_count/len(machine_stats)*100:.1f}%)")

# ============================================================================
# 8. 生成控制台报告和 CSV 文件
# ============================================================================
print("\n[8/8] 生成统计报告...")

print("\n" + "=" * 80)
print("📊 整体统计概况")
print("=" * 80)

print(f"\n【机器数量】")
print(f"  总机器数: {len(machine_stats)}")
print(f"  ✅ 适合进模型: {suitable_count} ({suitable_count/len(machine_stats)*100:.1f}%)")
print(f"  ⚠️ 不适合进模型: {unsuitable_count} ({unsuitable_count/len(machine_stats)*100:.1f}%)")

print(f"\n【活跃度统计】")
print(f"  平均活跃天数: {machine_stats['active_days'].mean():.1f} 天")
print(f"  平均总天数: {machine_stats['total_days'].mean():.1f} 天")
print(f"  平均活跃度比例: {machine_stats['active_days_ratio'].mean():.2%}")
print(f"  中位活跃度比例: {machine_stats['active_days_ratio'].median():.2%}")

print(f"\n【出杯统计】")
print(f"  平均日均出杯: {machine_stats['mean_cups_cnt'].mean():.1f} 杯")
print(f"  中位日均出杯: {machine_stats['mean_cups_cnt'].median():.1f} 杯")
print(f"  平均标准差: {machine_stats['std_cups_cnt'].mean():.1f}")
print(f"  平均变异系数: {machine_stats['cv_cups'].mean():.2f}")

print(f"\n【错误/清洗统计】")
print(f"  平均错误次数: {machine_stats['total_errors'].mean():.1f}")
print(f"  平均清洗次数: {machine_stats['total_cleanings'].mean():.1f}")
print(f"  有错误记录的机器: {(machine_stats['total_errors'] > 0).sum()} 台")
print(f"  有清洗记录的机器: {(machine_stats['total_cleanings'] > 0).sum()} 台")

print("\n" + "=" * 80)
print("❌ 不适合进模型的机器问题分析")
print("=" * 80)

unsuitable_machines = machine_stats[~machine_stats['is_suitable']]
if len(unsuitable_machines) > 0:
    issue_counts = unsuitable_machines['issues'].value_counts()
    print(f"\n问题分布:")
    for issue, count in issue_counts.head(10).items():
        print(f"  • {issue}: {count} 台 ({count/len(unsuitable_machines)*100:.1f}%)")
    
    print(f"\n示例不适合的机器 (前10台):")
    cols_to_show = ['external_id', 'active_days_ratio', 'mean_cups_cnt', 'std_cups_cnt', 'cv_cups', 'issues']
    print(unsuitable_machines[cols_to_show].head(10).to_string(index=False))
else:
    print("\n所有机器都适合进模型！")

print("\n" + "=" * 80)
print("✅ 适合进模型的机器概况 (前10台)")
print("=" * 80)

suitable_machines = machine_stats[machine_stats['is_suitable']]
if len(suitable_machines) > 0:
    cols_to_show = ['external_id', 'active_days', 'active_days_ratio', 'mean_cups_cnt', 
                    'std_cups_cnt', 'cv_cups', 'total_errors', 'total_cleanings']
    print(suitable_machines[cols_to_show].head(10).to_string(index=False))
else:
    print("\n⚠️ 没有机器适合进模型")

# 保存到 CSV
output_file = 'machine_quality_stats.csv'
machine_stats.to_csv(output_file, index=False)
print(f"\n" + "=" * 80)
print(f"💾 统计结果已保存到: {output_file}")
print("=" * 80)

print("\n" + "=" * 80)
print("📋 结论与建议")
print("=" * 80)

print(f"\n【数据质量评估】")
if suitable_count / len(machine_stats) >= 0.7:
    print(f"  ✅ 数据质量良好！{suitable_count/len(machine_stats)*100:.1f}% 的机器适合进模型")
elif suitable_count / len(machine_stats) >= 0.5:
    print(f"  ⚠️ 数据质量中等。{suitable_count/len(machine_stats)*100:.1f}% 的机器适合进模型")
else:
    print(f"  ❌ 数据质量较差！仅 {suitable_count/len(machine_stats)*100:.1f}% 的机器适合进模型")

print(f"\n【建议】")
print(f"  1. 使用 {suitable_count} 台适合的机器进行 Isolation Forest 模型训练")
print(f"  2. 对不适合的机器，可以考虑：")
print(f"     - 等待累积更多数据后再纳入")
print(f"     - 排除测试机器（如 RETURNED 开头的）")
print(f"     - 单独分析极度不稳定的机器")
print(f"  3. 后续特征工程时，优先使用适合的机器数据")

print("\n" + "=" * 80)
print("✨ 统计分析完成！")
print("=" * 80)

