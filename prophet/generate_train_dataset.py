import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# ==================== 配置参数 ====================
# 训练数据和测试数据分割日期
TRAIN_END_DATE = datetime(2026, 1, 20)  # 训练数据截止日期（包含）
VALIDATE_START_DATE = datetime(2026, 1, 21)  # 验证数据开始日期（包含）
VALIDATE_END_DATE = datetime(2026, 2, 4)  # 验证数据截止日期（包含）

# 数据筛选条件
MIN_DATA_DAYS = 70  # 最少数据天数要求（入选条件）

# 异常值检测参数
IQR_MULTIPLIER = 2.5  # IQR倍数，用于确定异常值边界
# ==================================================

# 获取脚本所在目录的父目录（项目根目录）
BASE_DIR = Path(__file__).resolve().parent.parent


def detect_and_smooth_outliers(df, iqr_multiplier=2.5):
    """
    使用IQR方法检测并平滑异常值
    
    参数:
    - df: DataFrame with 'ds' and 'y' columns
    - iqr_multiplier: IQR倍数，用于确定异常值边界（默认2.5）
    
    返回:
    - df_cleaned: 平滑后的数据
    - num_outliers: 异常值数量
    """
    df = df.copy()
    
    # 计算IQR（四分位距）
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    
    # 定义异常值边界
    # 注意：对于用量数据，0值（不使用）是正常的，只检测上界异常值
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    # 标记异常值（只标记过高的值，不标记0和低值）
    # 原因：0表示正常断档（如周末不使用），低值也是正常的
    outlier_mask = (df['y'] > upper_bound)
    num_outliers = outlier_mask.sum()
    
    if num_outliers > 0:
        # 对每个异常值，用前28天的非异常值中位数替代
        for idx in df[outlier_mask].index:
            # 获取前28天的数据（不包括当前点）
            window_start = max(0, idx - 28)
            window_data = df.loc[window_start:idx-1, 'y']
            
            # 只使用正常值（非异常且大于0）
            normal_values = window_data[
                (window_data <= upper_bound) & 
                (window_data >= lower_bound) & 
                (window_data > 0)
            ]
            
            if len(normal_values) > 0:
                # 用中位数替代（对极端值更鲁棒）
                replacement_value = int(normal_values.median())
            else:
                # 如果没有正常值，用所有非异常值的中位数
                all_normal = df[~outlier_mask]['y']
                replacement_value = int(all_normal.median()) if len(all_normal) > 0 else int(Q1)
            
            # 替换值
            df.loc[idx, 'y'] = replacement_value
    
    return df, num_outliers

# 读取数据
df = pd.read_csv(BASE_DIR / 'data/preprocessing_data/Result_bean1.csv')

# 将ds列转换为datetime格式
df['ds'] = pd.to_datetime(df['ds'])

print(f"读取数据完成，共 {len(df)} 条记录")
print(f"数据日期范围: {df['ds'].min()} 到 {df['ds'].max()}")
print(f"唯一的 customer_id 数量: {df['customer_id'].nunique()}")

print("\n配置参数:")
print(f"  训练数据截止日期: {TRAIN_END_DATE.strftime('%Y-%m-%d')}")
print(f"  验证数据日期范围: {VALIDATE_START_DATE.strftime('%Y-%m-%d')} 至 {VALIDATE_END_DATE.strftime('%Y-%m-%d')}")
print(f"  最少数据天数要求: {MIN_DATA_DAYS}天")
print(f"  异常值检测IQR倍数: {IQR_MULTIPLIER}")

# 步骤1: 筛选条件 - TRAIN_END_DATE前有数据 且 VALIDATE_START_DATE到VALIDATE_END_DATE有数据
# 找出TRAIN_END_DATE前有数据的customer_id
customer_before_cutoff = df[df['ds'] <= TRAIN_END_DATE]['customer_id'].unique()
print(f"\n步骤1: {TRAIN_END_DATE.strftime('%Y-%m-%d')}前有数据的customer: {len(customer_before_cutoff)} 个")

# 找出VALIDATE_START_DATE到VALIDATE_END_DATE有数据的customer_id
customer_in_validate_period = df[
    (df['ds'] >= VALIDATE_START_DATE) & 
    (df['ds'] <= VALIDATE_END_DATE)
]['customer_id'].unique()
print(f"        {VALIDATE_START_DATE.strftime('%Y-%m-%d')}到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}有数据的customer: {len(customer_in_validate_period)} 个")

# 同时满足两个条件的customer_id
valid_date_customer = set(customer_before_cutoff) & set(customer_in_validate_period)
print(f"        同时满足两个条件的customer: {len(valid_date_customer)} 个")

# 只保留这些customer_id的所有数据
df_filtered = df[df['customer_id'].isin(valid_date_customer)].copy()

# 步骤2: 为满足日期条件的所有customer_id补0到VALIDATE_END_DATE
# 找到每个customer_id的第一天
customer_date_ranges = df_filtered.groupby('customer_id')['ds'].agg(['min']).reset_index()
customer_date_ranges.columns = ['customer_id', 'first_date']

# 创建一个空的DataFrame来存储补全后的数据
filled_data = []

for _, row in customer_date_ranges.iterrows():
    customer_id = row['customer_id']
    first_date = row['first_date']
    
    # 获取该customer_id的所有数据
    customer_data = df_filtered[df_filtered['customer_id'] == customer_id].copy()
    
    # 创建完整的日期范围（从第一天到VALIDATE_END_DATE）
    date_range = pd.date_range(start=first_date, end=VALIDATE_END_DATE, freq='D')
    
    # 创建一个完整的DataFrame
    complete_df = pd.DataFrame({'ds': date_range})
    complete_df['customer_id'] = customer_id
    
    # 将实际数据与完整日期范围合并
    # 先按日期聚合（如果同一天有多条记录，求和）
    customer_data_agg = customer_data.groupby('ds')['y'].sum().reset_index()
    
    # 合并
    merged = complete_df.merge(customer_data_agg, on='ds', how='left')
    
    # 填充缺失值
    merged['y'] = merged['y'].fillna(0)
    
    filled_data.append(merged)

# 合并所有补全后的数据
df_filled = pd.concat(filled_data, ignore_index=True)
print(f"\n步骤2: 完成所有满足日期条件的设备数据补0（补到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}）")

# 步骤3: 在补0后的数据基础上，筛选出TRAIN_END_DATE之前有超过MIN_DATA_DAYS天真实数据的customer_id
# 🔥 关键修改：只统计TRAIN_END_DATE之前的真实数据天数（y > 0）
df_train_period = df_filtered[df_filtered['ds'] <= TRAIN_END_DATE].copy()
customer_train_days = df_train_period.groupby('customer_id')['ds'].nunique().reset_index(name='train_days')
valid_customer = customer_train_days[customer_train_days['train_days'] >= MIN_DATA_DAYS]['customer_id'].values
invalid_customer = customer_train_days[customer_train_days['train_days'] < MIN_DATA_DAYS]['customer_id'].values
print(f"\n步骤3: 在补0后的数据中，筛选{TRAIN_END_DATE.strftime('%Y-%m-%d')}前有>={MIN_DATA_DAYS}天真实数据的设备")
print(f"        入选设备（>={MIN_DATA_DAYS}天真实数据）: {len(valid_customer)} 个")
print(f"        落选设备（<{MIN_DATA_DAYS}天真实数据）: {len(invalid_customer)} 个")

# 打印一些统计信息
if len(invalid_customer) > 0:
    print(f"        数据天数不足{MIN_DATA_DAYS}天的设备示例: ")
    filtered_out = customer_train_days[customer_train_days['train_days'] < MIN_DATA_DAYS]
    for _, row in filtered_out.head(5).iterrows():
        print(f"          customer_{row['customer_id']}: {row['train_days']}天（训练期数据）")

# 保存入选设备的补0后数据
df_selected = df_filled[df_filled['customer_id'].isin(valid_customer)].copy()
# 保存落选设备的补0后数据
df_unselected = df_filled[df_filled['customer_id'].isin(invalid_customer)].copy()

# 步骤4: 为入选的customer_id生成三个数据集（已经补0完成）
df_final = df_selected.copy()
print(f"\n步骤4: 准备生成入选设备的三个数据集（数据已补0到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}）")

# 步骤4: 为入选的customer_id生成三个数据集
# 4.1 batch_train_data：从开始到TRAIN_END_DATE，检测并平滑激增数据
# 4.2 batch_train_validate：VALIDATE_START_DATE到VALIDATE_END_DATE（真实数据）
# 4.3 batch_train_whole_data：所有真实数据补0到VALIDATE_END_DATE
train_output_dir = BASE_DIR / 'data/batch_train_data'
validate_output_dir = BASE_DIR / 'data/batch_train_validate'
whole_train_output_dir = BASE_DIR / 'data/batch_train_whole_data'

# 创建输出目录
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(validate_output_dir, exist_ok=True)
os.makedirs(whole_train_output_dir, exist_ok=True)

# 清空目录中的旧文件
for output_dir in [train_output_dir, validate_output_dir, whole_train_output_dir]:
    for file in os.listdir(output_dir):
        if file.endswith('.csv'):
            os.remove(os.path.join(output_dir, file))

# 统计异常值信息
total_outliers_smoothed = 0
customer_with_outliers = 0

for customer_id in valid_customer:
    # 获取该customer_id的数据
    customer_data = df_final[df_final['customer_id'] == customer_id].copy()
    
    # 获取第一个数据日期
    first_date = customer_data['ds'].min()
    
    # 聚合数据（如果有重复日期）
    customer_data_agg = customer_data.groupby('ds')['y'].sum().reset_index()
    
    # 🔥 异常值检测和平滑（只对TRAIN_END_DATE及之前的数据）
    # 分割数据：训练期数据 vs 验证期数据
    train_period_mask = customer_data_agg['ds'] <= TRAIN_END_DATE
    train_period_data = customer_data_agg[train_period_mask].copy()
    validate_period_data = customer_data_agg[~train_period_mask].copy()
    
    # 只对训练期数据进行异常值检测和平滑
    train_period_cleaned, num_outliers = detect_and_smooth_outliers(train_period_data, iqr_multiplier=IQR_MULTIPLIER)
    
    # 统计异常值
    if num_outliers > 0:
        customer_with_outliers += 1
        total_outliers_smoothed += num_outliers
    
    # 4.1 生成batch_train_data文件（从开始到TRAIN_END_DATE，已平滑）
    train_data = train_period_cleaned.copy()
    train_data['y'] = train_data['y'].astype(int)
    train_output_file = os.path.join(train_output_dir, f'customer_{customer_id}.csv')
    train_data.to_csv(train_output_file, index=False)
    
    # 4.2 生成batch_train_validate文件（VALIDATE_START_DATE到VALIDATE_END_DATE，真实数据）
    validate_data = validate_period_data.copy()
    validate_data['y'] = validate_data['y'].astype(int)
    validate_output_file = os.path.join(validate_output_dir, f'customer_{customer_id}.csv')
    validate_data.to_csv(validate_output_file, index=False)
    
    # 4.3 生成batch_train_whole_data文件（所有真实数据补0到VALIDATE_END_DATE）
    whole_data = customer_data_agg.copy()
    whole_data['y'] = whole_data['y'].astype(int)
    whole_output_file = os.path.join(whole_train_output_dir, f'customer_{customer_id}.csv')
    whole_data.to_csv(whole_output_file, index=False)

print(f"\n步骤5: 成功生成入选设备的三个数据集")
print(f"  - batch_train_data: {len(valid_customer)} 个文件（已平滑激增数据）")
print(f"  - batch_train_validate: {len(valid_customer)} 个文件（{VALIDATE_START_DATE.strftime('%Y-%m-%d')}到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}真实数据）")
print(f"  - batch_train_whole_data: {len(valid_customer)} 个文件（所有真实数据补0到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}）")
print(f"  - 异常值平滑: {customer_with_outliers} 个设备，共 {total_outliers_smoothed} 个异常值被平滑")
print(f"  - 注意: 异常值检测仅应用于{TRAIN_END_DATE.strftime('%Y-%m-%d')}及之前的数据")

# 步骤6: 为落选的customer_id生成数据（已经补0到VALIDATE_END_DATE）
untrain_output_dir = BASE_DIR / 'data/batch_untrain_data'

# 创建输出目录
os.makedirs(untrain_output_dir, exist_ok=True)

# 清空目录中的旧文件
for file in os.listdir(untrain_output_dir):
    if file.endswith('.csv'):
        os.remove(os.path.join(untrain_output_dir, file))

for customer_id in invalid_customer:
    # 获取该customer_id的补0后数据（已经在步骤2中补0完成）
    customer_data = df_unselected[df_unselected['customer_id'] == customer_id].copy()
    
    # 按日期聚合（如果同一天有多条记录，求和）
    customer_data_agg = customer_data.groupby('ds')['y'].sum().reset_index()
    customer_data_agg['y'] = customer_data_agg['y'].astype(int)
    
    # 保存完整数据文件
    output_file = os.path.join(untrain_output_dir, f'customer_{customer_id}.csv')
    customer_data_agg.to_csv(output_file, index=False)

print(f"\n步骤6: 成功生成落选设备的数据")
print(f"  - batch_untrain_data: {len(invalid_customer)} 个文件")
print(f"  - 处理方式: 已补0到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}，不做异常值处理")

# 打印一些统计信息
print("\n统计信息:")
print(f"- 原始数据中的customer总数: {df['customer_id'].nunique()}")
print(f"- 满足日期条件的customer数量: {len(valid_date_customer)}")
print(f"  （{TRAIN_END_DATE.strftime('%Y-%m-%d')}前有数据 且 {VALIDATE_START_DATE.strftime('%Y-%m-%d')}到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}有数据）")
print(f"- 入选的customer数量（训练期>={MIN_DATA_DAYS}天数据）: {len(valid_customer)}")
print(f"- 落选的customer数量（训练期<{MIN_DATA_DAYS}天数据）: {len(invalid_customer)}")
print(f"- 原始数据日期范围: {df['ds'].min()} 到 {df['ds'].max()}")
print(f"- 筛选条件:")
print(f"  1. {TRAIN_END_DATE.strftime('%Y-%m-%d')}前有数据")
print(f"  2. {VALIDATE_START_DATE.strftime('%Y-%m-%d')}到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}有数据")
print(f"  3. {TRAIN_END_DATE.strftime('%Y-%m-%d')}前有>={MIN_DATA_DAYS}天真实数据记录（入选）")
print(f"- 数据输出:")
print(f"  batch_train_data: 入选设备训练数据（到{TRAIN_END_DATE.strftime('%Y-%m-%d')}，已平滑激增）")
print(f"  batch_train_validate: 入选设备验证数据（{VALIDATE_START_DATE.strftime('%Y-%m-%d')}到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}）")
print(f"  batch_train_whole_data: 入选设备完整数据（补0到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}）")
print(f"  batch_untrain_data: 落选设备数据（补0到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}）")

# 打印所有有效的 customer_id 列表
sorted_customer_ids = sorted(valid_customer)
print(f"\n所有入选的 customer_id 列表 ({len(sorted_customer_ids)} 个):")
print(sorted_customer_ids)

# 展示一个示例文件的前几行和最后几行
if len(valid_customer) > 0:
    sample_id = sorted(valid_customer)[0]
    
    # 展示训练数据
    train_sample_file = os.path.join(train_output_dir, f'customer_{sample_id}.csv')
    train_sample_data = pd.read_csv(train_sample_file)
    print(f"\nbatch_train_data示例 customer_{sample_id}.csv:")
    print(f"  前5行:")
    print(train_sample_data.head(5).to_string(index=False))
    print(f"  最后5行:")
    print(train_sample_data.tail(5).to_string(index=False))
    print(f"  总共有 {len(train_sample_data)} 行数据（到{TRAIN_END_DATE.strftime('%Y-%m-%d')}）")
    
    # 展示验证数据
    validate_sample_file = os.path.join(validate_output_dir, f'customer_{sample_id}.csv')
    validate_sample_data = pd.read_csv(validate_sample_file)
    print(f"\nbatch_train_validate示例 customer_{sample_id}.csv:")
    print(f"  前5行:")
    print(validate_sample_data.head(5).to_string(index=False))
    print(f"  最后5行:")
    print(validate_sample_data.tail(5).to_string(index=False))
    print(f"  总共有 {len(validate_sample_data)} 行数据（{VALIDATE_START_DATE.strftime('%Y-%m-%d')}到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}）")
    
    # 展示whole_data
    whole_sample_file = os.path.join(whole_train_output_dir, f'customer_{sample_id}.csv')
    whole_sample_data = pd.read_csv(whole_sample_file)
    print(f"\nbatch_train_whole_data示例 customer_{sample_id}.csv:")
    print(f"  前5行:")
    print(whole_sample_data.head(5).to_string(index=False))
    print(f"  最后5行:")
    print(whole_sample_data.tail(5).to_string(index=False))
    print(f"  总共有 {len(whole_sample_data)} 行数据（补0到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}）")
    
    # 展示落选设备数据
    if len(invalid_customer) > 0:
        untrain_sample_id = sorted(invalid_customer)[0]
        untrain_sample_file = os.path.join(untrain_output_dir, f'customer_{untrain_sample_id}.csv')
        untrain_sample_data = pd.read_csv(untrain_sample_file)
        print(f"\nbatch_untrain_data示例 customer_{untrain_sample_id}.csv:")
        print(f"  前5行:")
        print(untrain_sample_data.head(5).to_string(index=False))
        print(f"  最后5行:")
        print(untrain_sample_data.tail(5).to_string(index=False))
        print(f"  总共有 {len(untrain_sample_data)} 行数据（补0到{VALIDATE_END_DATE.strftime('%Y-%m-%d')}）")
        print(f"  说明: 落选设备（训练期真实数据<{MIN_DATA_DAYS}天）")

print("\n✅ 所有数据处理完成！")
