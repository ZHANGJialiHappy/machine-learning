import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 测试日期配置 - 放在最前面
TEST_START_DATE = '2026-01-21'  
TEST_END_DATE = '2026-02-04'


# 获取脚本所在目录的父目录（项目根目录）
BASE_DIR = Path(__file__).resolve().parent.parent

# 记录总执行时间
script_start_time = time.time()

print("=" * 80)
print("4种预测方法精确度对比分析")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"测试起始日期: {TEST_START_DATE} (包含此日期)")

# 定义预测函数
def exponential_smoothing(history, alpha=0.3):
    """简单指数平滑"""
    if len(history) == 0:
        return 0
    smoothed = [history[0]]
    for val in history[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return smoothed[-1]

def weighted_moving_average(history_28):
    """指数衰减加权平均，近期权重更高"""
    if len(history_28) == 0:
        return 0
    # 指数衰减权重
    days = np.arange(len(history_28))
    weights = np.exp(-days / 7)[::-1]  # 反转使最近的权重最大
    weights = weights / weights.sum()  # 归一化
    return np.dot(history_28, weights)

print("\n预测方法:")
print("  1. 28天移动平均 (MA28)")
print("  2. 28天中位数P50 (Median)")
print("  3. 移动加权平均 (Weighted MA)")
print("  4. 指数平滑 (Exponential Smoothing)")

# 1. 扫描数据文件
print("\n[1/5] 扫描数据文件...")
untrain_data_dir = BASE_DIR / 'data/batch_untrain_data'

data_files = sorted(untrain_data_dir.glob('customer_*.csv'))
print(f"✓ 找到 {len(data_files)} 个设备数据文件")

# 读取第一个文件确定实际的测试日期范围
first_file = data_files[0]
first_df = pd.read_csv(first_file)
first_df['ds'] = pd.to_datetime(first_df['ds'])

# 获取测试期间的数据
test_start = pd.to_datetime(TEST_START_DATE)
test_end = pd.to_datetime(TEST_END_DATE)
test_data = first_df[(first_df['ds'] >= test_start) & (first_df['ds'] <= test_end)]
test_days = len(test_data)

print(f"✓ 测试日期范围: {test_start.strftime('%Y-%m-%d')} 至 {test_end.strftime('%Y-%m-%d')}")
print(f"✓ 测试天数: {test_days} 天")

# 2. 创建输出目录
print("\n[2/5] 创建输出目录...")
statistics_dir = BASE_DIR / 'statistics'
os.makedirs(statistics_dir, exist_ok=True)
print(f"✓ 目录确认: statistics/")

# 3. 批量对比预测方法
print("\n[3/5] 开始批量对比预测方法...")
print("-" * 80)

results = []
success_count = 0
failed_count = 0
failed_customers = []

for idx, data_path in enumerate(data_files, 1):
    # 提取设备ID
    customer_id = data_path.stem.replace('customer_', '')
    
    try:
        # 加载数据
        df = pd.read_csv(data_path)
        df['ds'] = pd.to_datetime(df['ds'])
        
        # 筛选测试期间的数据
        test_df = df[(df['ds'] >= test_start) & (df['ds'] <= test_end)].copy()
        
        if len(test_df) == 0:
            print(f"[{idx}/{len(data_files)}] ⚠️  跳过 customer_{customer_id}: 没有测试期间数据")
            failed_count += 1
            failed_customers.append((customer_id, "没有测试期间数据"))
            continue
        
        # 初始化各方法的预测结果
        ma28_predictions = []
        median_predictions = []
        weighted_predictions = []
        ets_predictions = []
        
        # 对测试期的每一天进行预测（使用该日期之前的历史数据）
        for _, row in test_df.iterrows():
            date = row['ds']
            
            # 获取该日期之前的所有历史数据（不包括该日期本身）
            history_df = df[df['ds'] < date]
            
            if len(history_df) == 0:
                # 没有历史数据，使用0作为预测值
                ma28_predictions.append(0)
                median_predictions.append(0)
                weighted_predictions.append(0)
                ets_predictions.append(0)
                continue
            
            # 获取前28天数据
            history_28 = history_df['y'].values[-28:]
            
            # 获取完整历史数据用于指数平滑
            full_history = history_df['y'].values
            
            # 方法1: 28天移动平均
            ma28_pred = np.mean(history_28) if len(history_28) > 0 else 0
            ma28_predictions.append(ma28_pred)
            
            # 方法2: 28天中位数(P50)
            median_pred = np.median(history_28) if len(history_28) > 0 else 0
            median_predictions.append(median_pred)
            
            # 方法3: 移动加权平均
            weighted_pred = weighted_moving_average(history_28)
            weighted_predictions.append(weighted_pred)
            
            # 方法4: 指数平滑
            ets_pred = exponential_smoothing(full_history) if len(full_history) > 0 else 0
            ets_predictions.append(ets_pred)
        
        # 计算各方法的评估指标
        actual_values = test_df['y'].values
        
        # MAE (Mean Absolute Error)
        ma28_mae = mean_absolute_error(actual_values, ma28_predictions)
        median_mae = mean_absolute_error(actual_values, median_predictions)
        weighted_mae = mean_absolute_error(actual_values, weighted_predictions)
        ets_mae = mean_absolute_error(actual_values, ets_predictions)
        
        # RMSE (Root Mean Squared Error)
        ma28_rmse = np.sqrt(mean_squared_error(actual_values, ma28_predictions))
        median_rmse = np.sqrt(mean_squared_error(actual_values, median_predictions))
        weighted_rmse = np.sqrt(mean_squared_error(actual_values, weighted_predictions))
        ets_rmse = np.sqrt(mean_squared_error(actual_values, ets_predictions))
        
        # MAPE (Mean Absolute Percentage Error) - 避免除以0
        def calculate_mape(y_true, y_pred):
            mask = y_true != 0
            if mask.sum() == 0:
                return 0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        ma28_mape = calculate_mape(actual_values, np.array(ma28_predictions))
        median_mape = calculate_mape(actual_values, np.array(median_predictions))
        weighted_mape = calculate_mape(actual_values, np.array(weighted_predictions))
        ets_mape = calculate_mape(actual_values, np.array(ets_predictions))
        
        # 计算实际值的统计信息
        actual_mean = np.mean(actual_values)
        actual_std = np.std(actual_values)
        actual_min = np.min(actual_values)
        actual_max = np.max(actual_values)
        
        # 找出最佳方法（基于MAE）
        mae_dict = {
            'MA28': ma28_mae,
            'Median': median_mae,
            'Weighted': weighted_mae,
            'ETS': ets_mae
        }
        best_method = min(mae_dict, key=mae_dict.get)
        best_mae = mae_dict[best_method]
        
        print(f"[{idx}/{len(data_files)}] ✓ customer_{customer_id:15s} | "
              f"MA28: {ma28_mae:6.2f} | Median: {median_mae:6.2f} | "
              f"Weighted: {weighted_mae:6.2f} | ETS: {ets_mae:6.2f} | "
              f"最佳: {best_method}")
        
        results.append({
            'customer_id': customer_id,
            'ma28_mae': ma28_mae,
            'median_mae': median_mae,
            'weighted_mae': weighted_mae,
            'ets_mae': ets_mae,
            'ma28_rmse': ma28_rmse,
            'median_rmse': median_rmse,
            'weighted_rmse': weighted_rmse,
            'ets_rmse': ets_rmse,
            'ma28_mape': ma28_mape,
            'median_mape': median_mape,
            'weighted_mape': weighted_mape,
            'ets_mape': ets_mape,
            'actual_mean': actual_mean,
            'actual_std': actual_std,
            'actual_min': actual_min,
            'actual_max': actual_max,
            'best_method': best_method,
            'best_mae': best_mae,
            'test_samples': len(test_df)
        })
        
        success_count += 1
        
    except Exception as e:
        print(f"[{idx}/{len(data_files)}] ✗ customer_{customer_id:15s} | 错误: {str(e)}")
        failed_count += 1
        failed_customers.append((customer_id, str(e)))

# 4. 保存详细对比结果
print("\n" + "-" * 80)
print("\n[4/5] 保存详细对比结果...")

results_df = pd.DataFrame(results)
comparison_csv_path = statistics_dir / 'four_methods_comparison.csv'
results_df.to_csv(comparison_csv_path, index=False, encoding='utf-8-sig')
print(f"✓ 详细对比结果已保存: {comparison_csv_path}")

# 5. 生成汇总统计报告
print("\n[5/5] 生成汇总统计报告...")

total_duration = time.time() - script_start_time

if len(results_df) > 0:
    # 计算各方法的统计信息
    ma28_stats = {
        'mae': results_df['ma28_mae'].describe(),
        'rmse': results_df['ma28_rmse'].describe(),
        'mape': results_df['ma28_mape'].describe()
    }
    median_stats = {
        'mae': results_df['median_mae'].describe(),
        'rmse': results_df['median_rmse'].describe(),
        'mape': results_df['median_mape'].describe()
    }
    weighted_stats = {
        'mae': results_df['weighted_mae'].describe(),
        'rmse': results_df['weighted_rmse'].describe(),
        'mape': results_df['weighted_mape'].describe()
    }
    ets_stats = {
        'mae': results_df['ets_mae'].describe(),
        'rmse': results_df['ets_rmse'].describe(),
        'mape': results_df['ets_mape'].describe()
    }
    
    # 统计最佳方法分布
    best_method_counts = results_df['best_method'].value_counts()
    
    # 定义MAE范围
    mae_ranges = [
        (0, 10, "优秀 (0-10g)"),
        (10, 20, "良好 (10-20g)"),
        (20, 50, "中等 (20-50g)"),
        (50, 100, "较差 (50-100g)"),
        (100, float('inf'), "很差 (>100g)")
    ]
    
    # 计算每种方法在各范围的分布
    method_columns = ['ma28_mae', 'median_mae', 'weighted_mae', 'ets_mae']
    method_names = ['MA28', 'Median', 'Weighted', 'ETS']
    
    distribution = {}
    for method_col, method_name in zip(method_columns, method_names):
        distribution[method_name] = []
        for min_val, max_val, label in mae_ranges:
            count = len(results_df[(results_df[method_col] >= min_val) & (results_df[method_col] < max_val)])
            percentage = count / len(results_df) * 100
            distribution[method_name].append((label, count, percentage))
    
    # 生成汇总报告文本
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("4种预测方法精确度对比报告")
    summary_lines.append("=" * 80)
    summary_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    summary_lines.append("【测试概览】")
    summary_lines.append(f"数据来源:        data/batch_untrain_data")
    summary_lines.append(f"测试日期范围:    {TEST_START_DATE} 至 {TEST_END_DATE}")
    summary_lines.append(f"测试天数:        {test_days} 天")
    summary_lines.append(f"总设备数:        {len(data_files)}")
    summary_lines.append(f"成功测试:        {success_count} ({success_count/len(data_files)*100:.1f}%)")
    summary_lines.append(f"失败/跳过:       {failed_count} ({failed_count/len(data_files)*100:.1f}%)")
    summary_lines.append("")
    summary_lines.append("【预测方法说明】")
    summary_lines.append("  1. 28天移动平均 (MA28): 使用前28天数据的简单平均值")
    summary_lines.append("  2. 28天中位数P50 (Median): 使用前28天数据的中位数")
    summary_lines.append("  3. 移动加权平均 (Weighted MA): 指数衰减加权，近期数据权重更高")
    summary_lines.append("  4. 指数平滑 (ETS): 简单指数平滑，alpha=0.3")
    summary_lines.append("")
    summary_lines.append("【MAE指标对比】（平均绝对误差，单位：克）")
    summary_lines.append("")
    summary_lines.append(f"{'方法':<20} {'平均值':>10} {'中位数':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'28天移动平均':<20} {ma28_stats['mae']['mean']:>10.2f} {ma28_stats['mae']['50%']:>10.2f} {ma28_stats['mae']['std']:>10.2f} {ma28_stats['mae']['min']:>10.2f} {ma28_stats['mae']['max']:>10.2f}")
    summary_lines.append(f"{'28天中位数P50':<20} {median_stats['mae']['mean']:>10.2f} {median_stats['mae']['50%']:>10.2f} {median_stats['mae']['std']:>10.2f} {median_stats['mae']['min']:>10.2f} {median_stats['mae']['max']:>10.2f}")
    summary_lines.append(f"{'移动加权平均':<20} {weighted_stats['mae']['mean']:>10.2f} {weighted_stats['mae']['50%']:>10.2f} {weighted_stats['mae']['std']:>10.2f} {weighted_stats['mae']['min']:>10.2f} {weighted_stats['mae']['max']:>10.2f}")
    summary_lines.append(f"{'指数平滑':<20} {ets_stats['mae']['mean']:>10.2f} {ets_stats['mae']['50%']:>10.2f} {ets_stats['mae']['std']:>10.2f} {ets_stats['mae']['min']:>10.2f} {ets_stats['mae']['max']:>10.2f}")
    summary_lines.append("")
    
    summary_lines.append("【RMSE指标对比】（均方根误差，单位：克）")
    summary_lines.append("")
    summary_lines.append(f"{'方法':<20} {'平均值':>10} {'中位数':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'28天移动平均':<20} {ma28_stats['rmse']['mean']:>10.2f} {ma28_stats['rmse']['50%']:>10.2f} {ma28_stats['rmse']['std']:>10.2f} {ma28_stats['rmse']['min']:>10.2f} {ma28_stats['rmse']['max']:>10.2f}")
    summary_lines.append(f"{'28天中位数P50':<20} {median_stats['rmse']['mean']:>10.2f} {median_stats['rmse']['50%']:>10.2f} {median_stats['rmse']['std']:>10.2f} {median_stats['rmse']['min']:>10.2f} {median_stats['rmse']['max']:>10.2f}")
    summary_lines.append(f"{'移动加权平均':<20} {weighted_stats['rmse']['mean']:>10.2f} {weighted_stats['rmse']['50%']:>10.2f} {weighted_stats['rmse']['std']:>10.2f} {weighted_stats['rmse']['min']:>10.2f} {weighted_stats['rmse']['max']:>10.2f}")
    summary_lines.append(f"{'指数平滑':<20} {ets_stats['rmse']['mean']:>10.2f} {ets_stats['rmse']['50%']:>10.2f} {ets_stats['rmse']['std']:>10.2f} {ets_stats['rmse']['min']:>10.2f} {ets_stats['rmse']['max']:>10.2f}")
    summary_lines.append("")
    
    summary_lines.append("【MAPE指标对比】（平均绝对百分比误差，单位：%）")
    summary_lines.append("")
    summary_lines.append(f"{'方法':<20} {'平均值':>10} {'中位数':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'28天移动平均':<20} {ma28_stats['mape']['mean']:>10.2f} {ma28_stats['mape']['50%']:>10.2f} {ma28_stats['mape']['std']:>10.2f} {ma28_stats['mape']['min']:>10.2f} {ma28_stats['mape']['max']:>10.2f}")
    summary_lines.append(f"{'28天中位数P50':<20} {median_stats['mape']['mean']:>10.2f} {median_stats['mape']['50%']:>10.2f} {median_stats['mape']['std']:>10.2f} {median_stats['mape']['min']:>10.2f} {median_stats['mape']['max']:>10.2f}")
    summary_lines.append(f"{'移动加权平均':<20} {weighted_stats['mape']['mean']:>10.2f} {weighted_stats['mape']['50%']:>10.2f} {weighted_stats['mape']['std']:>10.2f} {weighted_stats['mape']['min']:>10.2f} {weighted_stats['mape']['max']:>10.2f}")
    summary_lines.append(f"{'指数平滑':<20} {ets_stats['mape']['mean']:>10.2f} {ets_stats['mape']['50%']:>10.2f} {ets_stats['mape']['std']:>10.2f} {ets_stats['mape']['min']:>10.2f} {ets_stats['mape']['max']:>10.2f}")
    summary_lines.append("")
    
    summary_lines.append("【精确度排名】（按平均MAE从低到高，数值越低越精确）")
    method_mae_avg = {
        '28天移动平均': ma28_stats['mae']['mean'],
        '28天中位数P50': median_stats['mae']['mean'],
        '移动加权平均': weighted_stats['mae']['mean'],
        '指数平滑': ets_stats['mae']['mean']
    }
    sorted_methods = sorted(method_mae_avg.items(), key=lambda x: x[1])
    for rank, (method, mae) in enumerate(sorted_methods, 1):
        summary_lines.append(f"{rank}. {method:<20} MAE: {mae:>8.2f}g")
    
    summary_lines.append("")
    summary_lines.append("【MAE性能分布对比】")
    summary_lines.append(f"{'MAE范围':<20} {'MA28':>12} {'Median':>12} {'Weighted':>12} {'ETS':>12}")
    summary_lines.append("-" * 80)
    for i in range(len(mae_ranges)):
        range_label = mae_ranges[i][2]
        ma28_dist = distribution['MA28'][i]
        median_dist = distribution['Median'][i]
        weighted_dist = distribution['Weighted'][i]
        ets_dist = distribution['ETS'][i]
        
        summary_lines.append(f"{range_label:<20} {ma28_dist[1]:>4}({ma28_dist[2]:>5.1f}%) "
                           f"{median_dist[1]:>4}({median_dist[2]:>5.1f}%) "
                           f"{weighted_dist[1]:>4}({weighted_dist[2]:>5.1f}%) "
                           f"{ets_dist[1]:>4}({ets_dist[2]:>5.1f}%)")
    
    summary_lines.append("")
    summary_lines.append("【最精确方法统计】（各设备上MAE最低的方法）")
    for method, count in best_method_counts.sort_index().items():
        percentage = count / len(results_df) * 100
        method_full_name = {
            'MA28': '28天移动平均',
            'Median': '28天中位数P50',
            'Weighted': '移动加权平均',
            'ETS': '指数平滑'
        }.get(method, method)
        summary_lines.append(f"{method_full_name:20s} 最优: {count:3d} 个设备 ({percentage:>5.1f}%)")
    
    summary_lines.append("")
    summary_lines.append("【执行性能】")
    summary_lines.append(f"总执行时间:     {total_duration:.2f}秒")
    summary_lines.append(f"平均每个设备:   {total_duration/len(data_files):.3f}秒")
    
    if failed_customers:
        summary_lines.append("")
        summary_lines.append(f"【失败/跳过的设备】({len(failed_customers)}个)")
        for customer_id, reason in failed_customers[:10]:
            summary_lines.append(f"  - customer_{customer_id}: {reason}")
        if len(failed_customers) > 10:
            summary_lines.append(f"  ... 还有 {len(failed_customers)-10} 个")
    
    summary_lines.append("")
    summary_lines.append("【关键发现】")
    best_overall = sorted_methods[0][0]
    worst_overall = sorted_methods[-1][0]
    summary_lines.append(f"1. 最精确方法: {best_overall} (平均MAE: {sorted_methods[0][1]:.2f}g)")
    summary_lines.append(f"2. 最差方法: {worst_overall} (平均MAE: {sorted_methods[-1][1]:.2f}g)")
    improvement = (sorted_methods[-1][1] - sorted_methods[0][1]) / sorted_methods[-1][1] * 100
    summary_lines.append(f"3. 精确度提升: {best_overall}比{worst_overall}的MAE降低 {improvement:.1f}%")
    
    # 找出最常见的最优方法
    most_common_best = best_method_counts.idxmax()
    most_common_best_full = {
        'MA28': '28天移动平均',
        'Median': '28天中位数P50',
        'Weighted': '移动加权平均',
        'ETS': '指数平滑'
    }.get(most_common_best, most_common_best)
    summary_lines.append(f"4. 最常见最优方法: {most_common_best_full} (在{best_method_counts[most_common_best]}个设备上表现最好)")
    
    summary_lines.append("")
    summary_lines.append("【输出文件】")
    summary_lines.append(f"详细对比结果: {comparison_csv_path}")
    summary_lines.append("")
    summary_lines.append("=" * 80)
    
    summary_text = "\n".join(summary_lines)
    
    # 保存汇总报告
    summary_path = statistics_dir / 'four_methods_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"✓ 汇总统计报告已保存: {summary_path}")
    
    # 打印汇总到控制台
    print("\n" + "=" * 80)
    print(summary_text)

else:
    print("⚠️  没有成功测试的设备")

print(f"\n脚本执行完成！总耗时: {total_duration:.2f}秒")
