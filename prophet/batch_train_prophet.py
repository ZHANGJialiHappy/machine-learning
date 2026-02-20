import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import os
from pathlib import Path
import time
from datetime import datetime


# 获取脚本所在目录的父目录（项目根目录）
BASE_DIR = Path(__file__).resolve().parent.parent

# 记录总执行时间
script_start_time = time.time()

print("=" * 80)
print("Prophet 批量训练 - customer 数据集")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("注: 训练数据已在预处理阶段进行异常值平滑")


# 2. 获取所有训练数据文件
print("\n[2/5] 扫描训练数据...")
batch_train_dir = BASE_DIR / 'data/batch_train_data'
all_files = sorted(batch_train_dir.glob('customer_*.csv'))
print(f"✓ 找到 {len(all_files)} 个训练数据文件")

# 3. 创建输出目录
print("\n[3/5] 创建输出目录...")
output_dir = BASE_DIR / 'batch_models'
os.makedirs(output_dir, exist_ok=True)
print(f"✓ 目录创建完成: batch_models/")

# 4. 批量训练模型
print("\n[4/5] 开始批量训练模型...")
print("-" * 80)

success_count = 0
failed_count = 0
failed_files = []

for idx, file_path in enumerate(all_files, 1):
    customer_name = file_path.stem  # 例如: customer_11
    
    try:
        # 加载数据（数据已在 generate_train_dataset.py 中进行异常值平滑）
        df = pd.read_csv(file_path)
        
        # 检查数据是否有效
        if len(df) < 2:
            print(f"[{idx}/{len(all_files)}] ⚠️  跳过 {customer_name}: 数据不足 (仅{len(df)}条)")
            failed_count += 1
            failed_files.append((customer_name, "数据不足"))
            continue
        
        # 配置 Prophet 模型（使用linear增长保持预测精度）
        model = Prophet(
            growth='linear',
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive',
            interval_width=0.8
        )
        
        # 训练模型
        train_start = time.time()
        model.fit(df)
        train_duration = time.time() - train_start
        
        # 保存模型
        model_path = output_dir / f'{customer_name}_model.json'
        with open(model_path, 'w') as f:
            f.write(model_to_json(model))
        
        model_size = os.path.getsize(model_path) / 1024  # KB
        
        print(f"[{idx}/{len(all_files)}] ✓ {customer_name:20s} | "
              f"数据: {len(df):4d}条 | "
              f"训练: {train_duration:5.2f}s | "
              f"模型: {model_size:6.1f}KB")
        
        success_count += 1
        
    except Exception as e:
        print(f"[{idx}/{len(all_files)}] ✗ {customer_name:20s} | 错误: {str(e)}")
        failed_count += 1
        failed_files.append((customer_name, str(e)))

# 5. 汇总统计
print("\n" + "-" * 80)
print("\n[5/5] 训练完成统计")
print("=" * 80)

total_duration = time.time() - script_start_time

print(f"\n总文件数:     {len(all_files)}")

if len(all_files) == 0:
    print("\n⚠️  错误：没有找到训练数据文件！")
    print("请先运行以下命令生成训练数据：")
    print("  python prophet/generate_train_dataset.py")
else:
    print(f"成功训练:     {success_count} ({success_count/len(all_files)*100:.1f}%)")
    print(f"失败/跳过:    {failed_count} ({failed_count/len(all_files)*100:.1f}%)")
    print(f"\n⏱️  总执行时间:  {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
    print(f"   平均每个模型: {total_duration/len(all_files):.2f}秒")

if success_count > 0:
    print(f"\n✓ 模型保存位置: {output_dir}/")
    print(f"  共 {success_count} 个模型文件")

if failed_files:
    print(f"\n⚠️  失败/跳过的文件列表 ({len(failed_files)}个):")
    for name, reason in failed_files[:10]:  # 只显示前10个
        print(f"  - {name}: {reason}")
    if len(failed_files) > 10:
        print(f"  ... 还有 {len(failed_files)-10} 个")

print("\n" + "=" * 80)
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
