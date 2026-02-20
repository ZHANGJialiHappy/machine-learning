import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

print("=" * 80)
print("Prophet模型被打败分析 (best_method != Prophet)")
print("=" * 80)

# 1. 读取性能对比数据
comparison_csv_path = BASE_DIR / 'statistics/prediction_methods_comparison.csv'
df = pd.read_csv(comparison_csv_path)

print(f"\n总设备数: {len(df)}")

# 2. 筛选出best_method不是Prophet的设备
poor_prophet_df = df[df['best_method'] != 'Prophet'].copy()
print(f"best_method不是Prophet的设备数: {len(poor_prophet_df)}")
print(f"占比: {len(poor_prophet_df)/len(df)*100:.1f}%")

# 3. 计算Prophet的MAPE (Mean Absolute Percentage Error)
# MAPE = (MAE / actual_mean) * 100%
# 为了避免除以0或非常小的值，只对actual_mean > 1的设备计算MAPE
poor_prophet_df['prophet_mape'] = np.where(
    poor_prophet_df['actual_mean'] > 1,  # 只对实际均值大于1g的设备计算MAPE
    (poor_prophet_df['prophet_mae'] / poor_prophet_df['actual_mean']) * 100,
    np.nan  # 对于用量极低的设备，设为NaN
)

# 4. 计算Prophet与最佳方法的MAE差距
# 需要根据best_method获取对应的MAE
def get_best_method_mae(row):
    method = row['best_method']
    if method == 'MA28':
        return row['ma28_mae']
    elif method == 'Median':
        return row['median_mae']
    elif method == 'Weighted':
        return row['weighted_mae']
    elif method == 'ETS':
        return row['ets_mae']
    else:
        return row['prophet_mae']

poor_prophet_df['best_method_mae'] = poor_prophet_df.apply(get_best_method_mae, axis=1)
poor_prophet_df['mae_gap'] = poor_prophet_df['prophet_mae'] - poor_prophet_df['best_method_mae']
poor_prophet_df['mae_gap_pct'] = (poor_prophet_df['mae_gap'] / poor_prophet_df['best_method_mae']) * 100

# 按MAE差距降序排列
poor_prophet_df = poor_prophet_df.sort_values('mae_gap', ascending=False)

print("\n" + "=" * 80)
print("Prophet被其他方法打败的设备列表")
print("=" * 80)
print(f"{'设备ID':<15} {'Prophet MAE':>12} {'最佳方法':>12} {'最佳MAE':>12} {'MAE差距':>12} {'差距%':>10}")
print("-" * 80)

for _, row in poor_prophet_df.iterrows():
    print(f"{row['customer_id']:<15} {row['prophet_mae']:>12.2f} {row['best_method']:>12} "
          f"{row['best_method_mae']:>12.2f} {row['mae_gap']:>12.2f} {row['mae_gap_pct']:>9.1f}%")

# 3. 分析这些设备的特征
print("\n" + "=" * 80)
print("Prophet被打败设备的统计特征")
print("=" * 80)

print(f"\n平均 Prophet MAE: {poor_prophet_df['prophet_mae'].mean():.2f}g")
print(f"平均 最佳方法 MAE: {poor_prophet_df['best_method_mae'].mean():.2f}g")
print(f"平均 MAE 差距: {poor_prophet_df['mae_gap'].mean():.2f}g ({poor_prophet_df['mae_gap_pct'].mean():.1f}%)")
print(f"平均实际用量: {poor_prophet_df['actual_mean'].mean():.2f}g")
print(f"平均标准差: {poor_prophet_df['actual_std'].mean():.2f}g")
print(f"变异系数 (CV): {(poor_prophet_df['actual_std'].mean() / poor_prophet_df['actual_mean'].mean()):.2f}")

# 4. 按最佳方法分组统计
print("\n" + "=" * 80)
print("按打败Prophet的方法分组")
print("=" * 80)
best_method_counts = poor_prophet_df['best_method'].value_counts()
print("\n各方法打败Prophet的次数:")
for method, count in best_method_counts.items():
    pct = count / len(poor_prophet_df) * 100
    avg_gap = poor_prophet_df[poor_prophet_df['best_method'] == method]['mae_gap'].mean()
    print(f"  {method:<12}: {count:>3}次 ({pct:>5.1f}%)  平均MAE优势: {avg_gap:.2f}g")

# 5. 对比Prophet与其他方法在这些设备上的表现
print("\n在这些设备上各方法的平均MAE:")
print(f"  Prophet:      {poor_prophet_df['prophet_mae'].mean():.2f}g")
print(f"  28天均值:     {poor_prophet_df['ma28_mae'].mean():.2f}g")
print(f"  28天中位数:   {poor_prophet_df['median_mae'].mean():.2f}g")
print(f"  加权平均:     {poor_prophet_df['weighted_mae'].mean():.2f}g")
print(f"  指数平滑:     {poor_prophet_df['ets_mae'].mean():.2f}g")

# 6. 分析异常值平滑的潜在影响
print("\n" + "=" * 80)
print("异常值平滑影响分析")
print("=" * 80)

print("\n【分析思路】")
print("异常值平滑可能导致Prophet表现差的原因：")
print("1. 过度平滑: 将真实的突变/趋势误判为异常值")
print("2. 破坏周期性: 平滑操作可能破坏数据的周期性模式")
print("3. 信息损失: 替换值可能不准确，导致模型学习错误的模式")
print("4. 边界效应: IQR方法可能对非正态分布数据不适用")

# 7. 检查具体设备的原始数据和训练数据
print("\n" + "=" * 80)
print("逐设备检查异常值平滑情况")
print("=" * 80)

train_data_dir = BASE_DIR / 'data/batch_train_data'
original_data_path = BASE_DIR / 'data/preprocessing_data/Result_66.csv'
original_df = pd.read_csv(original_data_path)
original_df['ds'] = pd.to_datetime(original_df['ds'])

customer_analysis = []

for _, row in poor_prophet_df.iterrows():
    customer_id = row['customer_id']
    
    # 读取训练数据（已平滑的）
    train_file = train_data_dir / f'customer_{customer_id}.csv'
    if not train_file.exists():
        print(f"⚠️  设备 {customer_id} 训练数据文件不存在")
        continue
    
    train_data = pd.read_csv(train_file)
    train_data['ds'] = pd.to_datetime(train_data['ds'])
    
    # 获取原始数据
    original_customer_data = original_df[original_df['customer_id'] == customer_id].copy()
    
    # 对原始数据按日期聚合（如果有重复）
    original_agg = original_customer_data.groupby('ds')['y'].sum().reset_index()
    
    # 合并训练数据和原始数据
    merged = train_data.merge(original_agg, on='ds', how='left', suffixes=('_smoothed', '_original'))
    merged['y_original'] = merged['y_original'].fillna(0)
    
    # 找出被修改的数据点
    merged['difference'] = merged['y_smoothed'] - merged['y_original']
    modified_points = merged[merged['difference'] != 0]
    
    # 计算统计信息
    num_modified = len(modified_points)
    if num_modified > 0:
        avg_original = modified_points['y_original'].mean()
        avg_smoothed = modified_points['y_smoothed'].mean()
        max_reduction = modified_points['difference'].min()
        max_original_value = modified_points['y_original'].max()
    else:
        avg_original = 0
        avg_smoothed = 0
        max_reduction = 0
        max_original_value = 0
    
    # 计算数据的基本统计特征
    non_zero_original = original_agg[original_agg['y'] > 0]['y']
    if len(non_zero_original) > 0:
        Q1 = non_zero_original.quantile(0.25)
        Q3 = non_zero_original.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 2.5 * IQR
    else:
        upper_bound = 0
    
    customer_analysis.append({
        'customer_id': customer_id,
        'prophet_mae': row['prophet_mae'],
        'best_method': row['best_method'],
        'best_method_mae': row['best_method_mae'],
        'mae_gap': row['mae_gap'],
        'mae_gap_pct': row['mae_gap_pct'],
        'actual_mean': row['actual_mean'],
        'num_modified_points': num_modified,
        'pct_modified': num_modified / len(train_data) * 100 if len(train_data) > 0 else 0,
        'avg_original_value': avg_original,
        'avg_smoothed_value': avg_smoothed,
        'max_reduction': max_reduction,
        'max_original_value': max_original_value,
        'upper_bound': upper_bound,
        'data_mean': original_agg['y'].mean(),
        'data_std': original_agg['y'].std(),
        'data_cv': original_agg['y'].std() / original_agg['y'].mean() if original_agg['y'].mean() > 0 else 0
    })
    
    if num_modified > 0:
        print(f"\n设备 {customer_id} (被{row['best_method']}打败):")
        print(f"  Prophet MAE: {row['prophet_mae']:.2f}g  vs  {row['best_method']}: {row['best_method_mae']:.2f}g  (差距: {row['mae_gap']:.2f}g)")
        print(f"  被修改的数据点: {num_modified} ({num_modified/len(train_data)*100:.1f}%)")
        print(f"  原始值平均: {avg_original:.2f}g -> 平滑后: {avg_smoothed:.2f}g")
        print(f"  最大削减: {abs(max_reduction):.2f}g (原值: {max_original_value:.2f}g)")
        print(f"  异常值上界: {upper_bound:.2f}g")
        print(f"  数据统计: 均值={original_agg['y'].mean():.2f}g, 标准差={original_agg['y'].std():.2f}g")
    else:
        print(f"\n设备 {customer_id} (被{row['best_method']}打败): 无异常值被平滑 (Prophet MAE={row['prophet_mae']:.2f}g vs {row['best_method']}: {row['best_method_mae']:.2f}g)")

# 8. 创建分析DataFrame
analysis_df = pd.DataFrame(customer_analysis)

# 9. 相关性分析
print("\n" + "=" * 80)
print("相关性分析: 异常值平滑与Prophet表现的关系")
print("=" * 80)

if len(analysis_df) > 0:
    correlation_modified = analysis_df[['mae_gap', 'mae_gap_pct', 'num_modified_points', 'pct_modified']].corr()
    print("\nProphet MAE差距 与异常值修改数量的相关系数:")
    print(f"  修改点数量: {correlation_modified.loc['mae_gap', 'num_modified_points']:.3f}")
    print(f"  修改点比例: {correlation_modified.loc['mae_gap', 'pct_modified']:.3f}")
    
    # 分组对比
    analysis_df['has_smoothing'] = analysis_df['num_modified_points'] > 0
    
    with_smoothing = analysis_df[analysis_df['has_smoothing']]
    without_smoothing = analysis_df[~analysis_df['has_smoothing']]
    
    print(f"\n分组对比:")
    print(f"  有异常值平滑的设备 ({len(with_smoothing)}个):")
    print(f"    平均 Prophet MAE: {with_smoothing['prophet_mae'].mean():.2f}g")
    print(f"    平均 最佳方法 MAE: {with_smoothing['best_method_mae'].mean():.2f}g")
    print(f"    平均 MAE 差距: {with_smoothing['mae_gap'].mean():.2f}g ({with_smoothing['mae_gap_pct'].mean():.1f}%)")
    print(f"    平均修改点数: {with_smoothing['num_modified_points'].mean():.1f}点")
    
    if len(without_smoothing) > 0:
        print(f"  无异常值平滑的设备 ({len(without_smoothing)}个):")
        print(f"    平均 Prophet MAE: {without_smoothing['prophet_mae'].mean():.2f}g")
        print(f"    平均 最佳方法 MAE: {without_smoothing['best_method_mae'].mean():.2f}g")
        print(f"    平均 MAE 差距: {without_smoothing['mae_gap'].mean():.2f}g ({without_smoothing['mae_gap_pct'].mean():.1f}%)")

# 10. 保存分析结果
output_path = BASE_DIR / 'statistics/poor_prophet_analysis.csv'
analysis_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✓ 分析结果已保存: {output_path}")

# 11. 生成可视化
print("\n生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prophet被其他方法打败的设备分析', fontsize=16, fontweight='bold')

# 图1: MAE差距分布
ax1 = axes[0, 0]
ax1.hist(poor_prophet_df['mae_gap'], bins=15, color='#e74c3c', alpha=0.7, edgecolor='black')
ax1.set_xlabel('MAE差距 (Prophet - 最佳方法, g)', fontsize=11)
ax1.set_ylabel('设备数量', fontsize=11)
ax1.set_title('Prophet MAE差距分布', fontsize=12, fontweight='bold')
ax1.axvline(poor_prophet_df['mae_gap'].mean(), color='navy', linestyle='--', linewidth=2, label=f'平均值: {poor_prophet_df["mae_gap"].mean():.2f}g')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: 各方法打败Prophet次数
ax2 = axes[0, 1]
best_method_counts = poor_prophet_df['best_method'].value_counts()
colors_map = {'MA28': '#3498db', 'Median': '#2ecc71', 'Weighted': '#f39c12', 'ETS': '#9b59b6'}
colors = [colors_map.get(method, '#95a5a6') for method in best_method_counts.index]
bars = ax2.bar(range(len(best_method_counts)), best_method_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(best_method_counts)))
ax2.set_xticklabels(best_method_counts.index, rotation=15)
ax2.set_ylabel('打败Prophet次数', fontsize=11)
ax2.set_title('各方法打败Prophet统计', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
# 添加数值标签
for i, (bar, count) in enumerate(zip(bars, best_method_counts.values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(count)}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 图3: 异常值修改比例 vs MAE差距
ax3 = axes[1, 0]
if len(analysis_df) > 0:
    ax3.scatter(analysis_df['pct_modified'], analysis_df['mae_gap'], 
                alpha=0.6, s=100, color='#9b59b6', edgecolor='black')
    ax3.set_xlabel('异常值修改比例 (%)', fontsize=11)
    ax3.set_ylabel('MAE差距 (Prophet - 最佳方法, g)', fontsize=11)
    ax3.set_title('异常值修改比例 vs Prophet MAE差距', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 添加趋势线
    if len(analysis_df) > 1:
        z = np.polyfit(analysis_df['pct_modified'], analysis_df['mae_gap'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(analysis_df['pct_modified'].min(), analysis_df['pct_modified'].max(), 100)
        ax3.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7, label=f'趋势线')
        ax3.legend()

# 图4: 数据变异系数 vs MAE差距
ax4 = axes[1, 1]
if len(analysis_df) > 0:
    ax4.scatter(analysis_df['data_cv'], analysis_df['mae_gap'], 
                alpha=0.6, s=100, color='#e67e22', edgecolor='black')
    ax4.set_xlabel('变异系数 (CV = std/mean)', fontsize=11)
    ax4.set_ylabel('MAE差距 (Prophet - 最佳方法, g)', fontsize=11)
    ax4.set_title('数据变异性 vs Prophet MAE差距', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = BASE_DIR / 'statistics/poor_prophet_analysis.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ 可视化图表已保存: {plot_path}")

# 12. 生成总结报告
print("\n" + "=" * 80)
print("【关键发现总结】")
print("=" * 80)

print("\n1. Prophet被打败的设备:")
print(f"   - 共 {len(poor_prophet_df)} 个设备 (best_method != Prophet)")
print(f"   - 平均 Prophet MAE: {poor_prophet_df['prophet_mae'].mean():.2f}g")
print(f"   - 平均 最佳方法 MAE: {poor_prophet_df['best_method_mae'].mean():.2f}g")
print(f"   - 平均 MAE 差距: {poor_prophet_df['mae_gap'].mean():.2f}g ({poor_prophet_df['mae_gap_pct'].mean():.1f}%)")
print(f"   - 平均实际用量: {poor_prophet_df['actual_mean'].mean():.2f}g")

print("\n2. 打败Prophet的方法分布:")
for method, count in best_method_counts.items():
    pct = count / len(poor_prophet_df) * 100
    avg_gap = poor_prophet_df[poor_prophet_df['best_method'] == method]['mae_gap'].mean()
    print(f"   - {method:<12}: {count:>3}次 ({pct:>5.1f}%)  平均优势: {avg_gap:.2f}g")

if len(analysis_df) > 0:
    print("\n3. 异常值平滑影响:")
    print(f"   - 有异常值平滑的设备: {len(analysis_df[analysis_df['has_smoothing']])} 个")
    print(f"   - 无异常值平滑的设备: {len(analysis_df[~analysis_df['has_smoothing']])} 个")
    
    corr_value = correlation_modified.loc['mae_gap', 'pct_modified']
    if abs(corr_value) < 0.3:
        correlation_strength = "弱相关"
    elif abs(corr_value) < 0.6:
        correlation_strength = "中等相关"
    else:
        correlation_strength = "强相关"
    
    print(f"   - 相关系数: {corr_value:.3f} ({correlation_strength})")
    
    if corr_value > 0.3:
        print("\n   ⚠️  异常值平滑比例越高，Prophet MAE差距越大，可能存在因果关系！")
    elif corr_value < -0.3:
        print("\n   ✓ 异常值平滑比例越高，Prophet MAE差距越小，平滑有正面效果")
    else:
        print("\n   ⚠️  异常值平滑与Prophet表现的相关性较弱，可能不是主要原因")

print("\n4. 建议:")
print("   - 对Prophet表现差的设备，考虑使用其最佳方法进行预测")
print("   - 检查异常值平滑逻辑是否过于激进")
print("   - 分析数据的实际特征（周期性、趋势性、突变等）")
print("   - 考虑调整Prophet模型参数（seasonality、changepoint等）")
print("   - 简单方法(MA28/Median/Weighted)在某些场景下可能更稳健")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
