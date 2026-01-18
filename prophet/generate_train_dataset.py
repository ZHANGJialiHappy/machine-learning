import pandas as pd


# 读取原始数据
df = pd.read_csv('data/train_data/caffee_consumption.csv')

# 将date列转换为日期格式
df['date'] = pd.to_datetime(df['date'])

# 按日期和设备ID分组，汇总每天每台机器的消耗量
daily_machine_data = df.groupby(['date', 'equipment_id'])['amount_g'].sum().reset_index()

# 计算每天的指标
daily_stats = daily_machine_data.groupby('date').apply(
    lambda x: pd.Series({
        'total_amount': x['amount_g'].sum(),  # 当天总消耗量
        'active_machines': (x['amount_g'] > 0).sum()  # amount_g不为0的机器数量
    })
).reset_index()

# 计算每天的平均消耗
# 如果active_machines为0，则平均消耗为0
daily_stats['daily_avg'] = daily_stats.apply(
    lambda row: row['total_amount'] / row['active_machines'] if row['active_machines'] > 0 else 0,
    axis=1
)

# 获取日期范围
min_date = daily_stats['date'].min()
max_date = daily_stats['date'].max()

# 创建完整的日期范围（包含所有日期）
date_range = pd.date_range(start=min_date, end=max_date, freq='D')
full_dates = pd.DataFrame({'date': date_range})

# 合并数据，缺失的日期会被填充
result = full_dates.merge(daily_stats[['date', 'daily_avg']], on='date', how='left')

# 将NaN填充为0
result['daily_avg'] = result['daily_avg'].fillna(0)

# 重命名列为ds和y
result.columns = ['ds', 'y']

# 将ds格式化为字符串格式 YYYY-MM-DD
result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')

# 将y转换为整数
result['y'] = result['y'].astype(int)

# 保存到CSV文件
output_path = 'data/train_data/daily_consumption.csv'
result.to_csv(output_path, index=False)

print(f"处理完成！共生成 {len(result)} 条记录")
print(f"日期范围：{result['ds'].min()} 至 {result['ds'].max()}")
print(f"\n前10条记录：")
print(result.head(10))
print(f"\n文件已保存至：{output_path}")
