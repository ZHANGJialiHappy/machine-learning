import pandas as pd
import numpy as np
from prophet.serialize import model_from_json
import os
from pathlib import Path
import time
from datetime import datetime
from sklearn.metrics import mean_absolute_error

# 获取脚本所在目录的父目录（项目根目录）
BASE_DIR = Path(__file__).resolve().parent.parent

# 记录总执行时间
script_start_time = time.time()

print("=" * 80)
print("预测方法性能对比分析")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 1. 扫描数据文件并确定测试日期范围
print("\n[1/6] 扫描数据文件并确定测试日期范围...")
models_dir = BASE_DIR / 'batch_models'
whole_data_dir = BASE_DIR / 'data/batch_train_whole_data'
validate_dir = BASE_DIR / 'data/batch_train_validate'

model_files = sorted(models_dir.glob('customer_*_model.json'))
print(f"✓ 找到 {len(model_files)} 个模型文件")

# 动态确定测试日期范围：从batch_train_validate中读取第一个文件获取测试日期
validate_files = sorted(validate_dir.glob('customer_*.csv'))
if len(validate_files) == 0:
    raise FileNotFoundError("未找到任何batch_train_validate文件")

# 读取第一个验证文件获取测试日期范围
first_validate_file = validate_files[0]
first_validate_df = pd.read_csv(first_validate_file)
first_validate_df['ds'] = pd.to_datetime(first_validate_df['ds'])

TEST_START_DATE = first_validate_df['ds'].min()
TEST_END_DATE = first_validate_df['ds'].max()

print(f"✓ 测试日期范围: {TEST_START_DATE.strftime('%Y-%m-%d')} 至 {TEST_END_DATE.strftime('%Y-%m-%d')}")
print(f"  (从 {first_validate_file.name} 中读取)")

print("\n对比方法:")
print(f"  1. Prophet模型 (使用batch_train_validate数据: {TEST_START_DATE.strftime('%Y-%m-%d')}至{TEST_END_DATE.strftime('%Y-%m-%d')})")
print(f"  2. 28天移动平均 (使用batch_train_whole_data: {TEST_START_DATE.strftime('%Y-%m-%d')}至{TEST_END_DATE.strftime('%Y-%m-%d')})")
print(f"  3. 28天中位数(P50) (使用batch_train_whole_data: {TEST_START_DATE.strftime('%Y-%m-%d')}至{TEST_END_DATE.strftime('%Y-%m-%d')})")
print(f"  4. 移动加权平均 (使用batch_train_whole_data: {TEST_START_DATE.strftime('%Y-%m-%d')}至{TEST_END_DATE.strftime('%Y-%m-%d')})")
print(f"  5. 指数平滑 (使用batch_train_whole_data: {TEST_START_DATE.strftime('%Y-%m-%d')}至{TEST_END_DATE.strftime('%Y-%m-%d')})")
print("\n注意: 其他方法对每个测试日期只使用该日期之前的历史数据进行预测")

# 2. 创建输出目录
print("\n[2/6] 创建输出目录...")
statistics_dir = BASE_DIR / 'statistics'
os.makedirs(statistics_dir, exist_ok=True)
print(f"✓ 目录确认: statistics/")

# 3. 定义预测函数
print("\n[3/6] 初始化预测函数...")

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
    days = np.arange(28)
    weights = np.exp(-days / 7)[::-1]  # 反转使最近的权重最大
    weights = weights / weights.sum()  # 归一化
    return np.dot(history_28, weights)

print("✓ 预测函数就绪")

# 4. 批量对比预测方法
print("\n[4/6] 开始批量对比预测方法...")
print("-" * 80)

results = []
success_count = 0
failed_count = 0
failed_customers = []

for idx, model_path in enumerate(model_files, 1):
    # 提取设备ID
    customer_id = model_path.stem.replace('customer_', '').replace('_model', '')
    
    # 检查对应的数据文件是否存在
    whole_data_path = whole_data_dir / f'customer_{customer_id}.csv'
    validate_path = validate_dir / f'customer_{customer_id}.csv'
    
    if not whole_data_path.exists() or not validate_path.exists():
        print(f"[{idx}/{len(model_files)}] ⚠️  跳过 customer_{customer_id}: 数据文件缺失")
        failed_count += 1
        failed_customers.append((customer_id, "数据文件缺失"))
        continue
    
    try:
        # 加载数据
        whole_df = pd.read_csv(whole_data_path)
        whole_df['ds'] = pd.to_datetime(whole_df['ds'])
        validate_df = pd.read_csv(validate_path)
        validate_df['ds'] = pd.to_datetime(validate_df['ds'])
        
        if len(validate_df) < 1:
            print(f"[{idx}/{len(model_files)}] ⚠️  跳过 customer_{customer_id}: 验证数据为空")
            failed_count += 1
            failed_customers.append((customer_id, "验证数据为空"))
            continue
        
        # 方法1: Prophet模型预测
        with open(model_path, 'r') as f:
            prophet_model = model_from_json(f.read())
        
        prophet_forecast = prophet_model.predict(validate_df[['ds']])
        
        # 后处理：将负数预测值截断为0（设备数据不应为负）
        prophet_forecast['yhat'] = prophet_forecast['yhat'].clip(lower=0)
        prophet_predictions = prophet_forecast['yhat'].values
        
        # 方法2-5: 从whole_data中提取测试期间数据
        test_start = TEST_START_DATE
        test_end = TEST_END_DATE
        
        # 从whole_data中筛选测试期间的数据
        test_df = whole_df[(whole_df['ds'] >= test_start) & (whole_df['ds'] <= test_end)].copy()
        
        if len(test_df) == 0:
            print(f"[{idx}/{len(model_files)}] ⚠️  跳过 customer_{customer_id}: whole_data中没有测试期间数据")
            failed_count += 1
            failed_customers.append((customer_id, "whole_data中没有测试期间数据"))
            continue
        
        # 初始化其他方法的预测结果
        ma28_predictions = []
        median_predictions = []
        weighted_predictions = []
        ets_predictions = []
        
        # 对测试期的每一天进行预测（使用该日期之前的历史数据）
        for _, row in test_df.iterrows():
            date = row['ds']
            
            # 获取该日期之前的所有历史数据（不包括该日期本身）
            history_df = whole_df[whole_df['ds'] < date]
            
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
            
            # 方法2: 28天移动平均
            ma28_pred = np.mean(history_28) if len(history_28) > 0 else 0
            ma28_predictions.append(ma28_pred)
            
            # 方法3: 28天中位数(P50)
            median_pred = np.median(history_28) if len(history_28) > 0 else 0
            median_predictions.append(median_pred)
            
            # 方法4: 移动加权平均
            if len(history_28) == 28:
                weighted_pred = weighted_moving_average(history_28)
            else:
                # 数据不足28天，使用简单平均
                weighted_pred = np.mean(history_28) if len(history_28) > 0 else 0
            weighted_predictions.append(weighted_pred)
            
            # 方法5: 指数平滑
            ets_pred = exponential_smoothing(full_history) if len(full_history) > 0 else 0
            ets_predictions.append(ets_pred)
        
        # 计算各方法的MAE（使用test_df中的实际值）
        actual_values = test_df['y'].values
        
        # 对齐Prophet的预测结果（确保Prophet也使用相同的测试日期）
        prophet_test_df = validate_df[validate_df['ds'].isin(test_df['ds'])].copy()
        prophet_forecast_filtered = prophet_forecast[prophet_forecast['ds'].isin(test_df['ds'])]
        prophet_predictions = prophet_forecast_filtered['yhat'].values
        
        prophet_mae = mean_absolute_error(actual_values, prophet_predictions)
        ma28_mae = mean_absolute_error(actual_values, ma28_predictions)
        median_mae = mean_absolute_error(actual_values, median_predictions)
        weighted_mae = mean_absolute_error(actual_values, weighted_predictions)
        ets_mae = mean_absolute_error(actual_values, ets_predictions)
        
        # 计算实际值的统计信息
        actual_mean = np.mean(actual_values)
        actual_std = np.std(actual_values)
        
        # 找出最佳方法
        mae_dict = {
            'Prophet': prophet_mae,
            'MA28': ma28_mae,
            'Median': median_mae,
            'Weighted': weighted_mae,
            'ETS': ets_mae
        }
        best_method = min(mae_dict, key=mae_dict.get)
        
        print(f"[{idx}/{len(model_files)}] ✓ customer_{customer_id:20s} | "
              f"Prophet: {prophet_mae:6.2f} | MA28: {ma28_mae:6.2f} | "
              f"Median: {median_mae:6.2f} | Weighted: {weighted_mae:6.2f} | "
              f"ETS: {ets_mae:6.2f} | 最佳: {best_method}")
        
        results.append({
            'customer_id': customer_id,
            'prophet_mae': prophet_mae,
            'ma28_mae': ma28_mae,
            'median_mae': median_mae,
            'weighted_mae': weighted_mae,
            'ets_mae': ets_mae,
            'actual_mean': actual_mean,
            'actual_std': actual_std,
            'best_method': best_method,
            'validate_samples': len(test_df)
        })
        
        success_count += 1
        
    except Exception as e:
        print(f"[{idx}/{len(model_files)}] ✗ customer_{customer_id:20s} | 错误: {str(e)}")
        failed_count += 1
        failed_customers.append((customer_id, str(e)))

# 5. 保存详细对比结果
print("\n" + "-" * 80)
print("\n[5/6] 保存详细对比结果...")

results_df = pd.DataFrame(results)
comparison_csv_path = statistics_dir / 'prediction_methods_comparison.csv'
results_df.to_csv(comparison_csv_path, index=False, encoding='utf-8-sig')
print(f"✓ 详细对比结果已保存: {comparison_csv_path}")

# 6. 生成汇总统计报告
print("\n[6/6] 生成汇总统计报告...")

total_duration = time.time() - script_start_time

if len(results_df) > 0:
    # 计算各方法的统计信息
    prophet_stats = results_df['prophet_mae'].describe()
    ma28_stats = results_df['ma28_mae'].describe()
    median_stats = results_df['median_mae'].describe()
    weighted_stats = results_df['weighted_mae'].describe()
    ets_stats = results_df['ets_mae'].describe()
    
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
    method_columns = ['prophet_mae', 'ma28_mae', 'median_mae', 'weighted_mae', 'ets_mae']
    method_names = ['Prophet', 'MA28', 'Median', 'Weighted', 'ETS']
    
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
    summary_lines.append("预测方法性能对比报告")
    summary_lines.append("=" * 80)
    summary_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    summary_lines.append("【验证概览】")
    summary_lines.append(f"总设备数:        {len(model_files)}")
    summary_lines.append(f"成功对比:        {success_count} ({success_count/len(model_files)*100:.1f}%)")
    summary_lines.append(f"失败/跳过:       {failed_count} ({failed_count/len(model_files)*100:.1f}%)")
    summary_lines.append(f"测试日期范围:    {TEST_START_DATE.strftime('%Y-%m-%d')} 至 {TEST_END_DATE.strftime('%Y-%m-%d')}")
    summary_lines.append(f"验证样本数:      {results_df['validate_samples'].iloc[0]}天/设备")
    summary_lines.append("")
    summary_lines.append("【测试数据说明】")
    summary_lines.append("  - Prophet模型: 使用batch_train_validate数据")
    summary_lines.append("  - 其他方法: 使用batch_train_whole_data中的测试期间数据")
    summary_lines.append("  - 其他方法对每个测试日期只使用该日期之前的历史数据")
    summary_lines.append("")
    summary_lines.append("【整体MAE统计】（平均绝对误差，单位：克）")
    summary_lines.append("")
    summary_lines.append(f"{'方法':<20} {'平均MAE':>10} {'中位数MAE':>12} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'Prophet模型':<20} {prophet_stats['mean']:>10.2f} {prophet_stats['50%']:>12.2f} {prophet_stats['std']:>10.2f} {prophet_stats['min']:>10.2f} {prophet_stats['max']:>10.2f}")
    summary_lines.append(f"{'28天均值':<20} {ma28_stats['mean']:>10.2f} {ma28_stats['50%']:>12.2f} {ma28_stats['std']:>10.2f} {ma28_stats['min']:>10.2f} {ma28_stats['max']:>10.2f}")
    summary_lines.append(f"{'28天中位数P50':<20} {median_stats['mean']:>10.2f} {median_stats['50%']:>12.2f} {median_stats['std']:>10.2f} {median_stats['min']:>10.2f} {median_stats['max']:>10.2f}")
    summary_lines.append(f"{'移动加权平均':<20} {weighted_stats['mean']:>10.2f} {weighted_stats['50%']:>12.2f} {weighted_stats['std']:>10.2f} {weighted_stats['min']:>10.2f} {weighted_stats['max']:>10.2f}")
    summary_lines.append(f"{'指数平滑':<20} {ets_stats['mean']:>10.2f} {ets_stats['50%']:>12.2f} {ets_stats['std']:>10.2f} {ets_stats['min']:>10.2f} {ets_stats['max']:>10.2f}")
    summary_lines.append("")
    summary_lines.append("【性能排名】（按平均MAE从低到高）")
    method_mae_avg = {
        'Prophet模型': prophet_stats['mean'],
        '28天均值': ma28_stats['mean'],
        '28天中位数P50': median_stats['mean'],
        '移动加权平均': weighted_stats['mean'],
        '指数平滑': ets_stats['mean']
    }
    sorted_methods = sorted(method_mae_avg.items(), key=lambda x: x[1])
    for rank, (method, mae) in enumerate(sorted_methods, 1):
        summary_lines.append(f"{rank}. {method:<20} MAE: {mae:>8.2f}g")
    
    summary_lines.append("")
    summary_lines.append("【性能分布对比】")
    summary_lines.append(f"{'MAE范围':<20} {'Prophet':>10} {'MA28':>10} {'Median':>10} {'Weighted':>10} {'ETS':>10}")
    summary_lines.append("-" * 80)
    for i in range(len(mae_ranges)):
        range_label = mae_ranges[i][2]
        prophet_dist = distribution['Prophet'][i]
        ma28_dist = distribution['MA28'][i]
        median_dist = distribution['Median'][i]
        weighted_dist = distribution['Weighted'][i]
        ets_dist = distribution['ETS'][i]
        
        summary_lines.append(f"{range_label:<20} {prophet_dist[1]:>4}({prophet_dist[2]:>5.1f}%) "
                           f"{ma28_dist[1]:>4}({ma28_dist[2]:>5.1f}%) "
                           f"{median_dist[1]:>4}({median_dist[2]:>5.1f}%) "
                           f"{weighted_dist[1]:>4}({weighted_dist[2]:>5.1f}%) "
                           f"{ets_dist[1]:>4}({ets_dist[2]:>5.1f}%)")
    
    summary_lines.append("")
    summary_lines.append("【最佳方法统计】（各设备上表现最好的方法）")
    for method, count in best_method_counts.items():
        percentage = count / len(results_df) * 100
        summary_lines.append(f"{method:15s} 最优: {count:3d} 个设备 ({percentage:>5.1f}%)")
    
    summary_lines.append("")
    summary_lines.append("【执行性能】")
    summary_lines.append(f"总执行时间:     {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
    summary_lines.append(f"平均每个设备:   {total_duration/len(model_files):.2f}秒")
    
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
    summary_lines.append(f"1. 最佳整体方法: {best_overall} (平均MAE: {sorted_methods[0][1]:.2f}g)")
    summary_lines.append(f"2. 最差整体方法: {sorted_methods[-1][0]} (平均MAE: {sorted_methods[-1][1]:.2f}g)")
    improvement = (sorted_methods[-1][1] - sorted_methods[0][1]) / sorted_methods[-1][1] * 100
    summary_lines.append(f"3. 性能提升: 最佳方法比最差方法MAE降低 {improvement:.1f}%")
    
    summary_lines.append("")
    summary_lines.append("【输出文件】")
    summary_lines.append(f"详细对比结果: {comparison_csv_path}")
    summary_lines.append("")
    summary_lines.append("=" * 80)
    
    summary_text = "\n".join(summary_lines)
    
    # 保存汇总报告
    summary_path = statistics_dir / 'methods_performance_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"✓ 汇总统计报告已保存: {summary_path}")
    
    # 打印汇总到控制台
    print("\n" + "=" * 80)
    print(summary_text)

else:
    print("⚠️  没有成功对比的设备")
