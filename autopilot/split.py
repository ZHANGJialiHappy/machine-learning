import pandas as pd
import os

# 创建输出目录
os.makedirs('data/model1', exist_ok=True)
os.makedirs('data/model2', exist_ok=True)

# 处理 cleaning-extract.csv
print("处理 cleaning-extract.csv...")
df_cleaning = pd.read_csv('data/cleaning-extract.csv')
df_cleaning[df_cleaning['equipment_model_id'] == 1].to_csv('data/model1/cleaning-extract.csv', index=False)
df_cleaning[df_cleaning['equipment_model_id'] == 2].to_csv('data/model2/cleaning-extract.csv', index=False)
print(f"  Model 1: {len(df_cleaning[df_cleaning['equipment_model_id'] == 1])} 条记录")
print(f"  Model 2: {len(df_cleaning[df_cleaning['equipment_model_id'] == 2])} 条记录")

# 处理 error-extract.csv
print("\n处理 error-extract.csv...")
df_error = pd.read_csv('data/error-extract.csv')
df_error[df_error['equipment_model_id'] == 1].to_csv('data/model1/error-extract.csv', index=False)
df_error[df_error['equipment_model_id'] == 2].to_csv('data/model2/error-extract.csv', index=False)
print(f"  Model 1: {len(df_error[df_error['equipment_model_id'] == 1])} 条记录")
print(f"  Model 2: {len(df_error[df_error['equipment_model_id'] == 2])} 条记录")

# 处理 usage-extract.xlsx
print("\n处理 usage-extract.xlsx...")
df_usage = pd.read_excel('data/usage-extract.xlsx')
df_usage[df_usage['equipment_model_id'] == 1].to_excel('data/model1/usage-extract.xlsx', index=False)
df_usage[df_usage['equipment_model_id'] == 2].to_excel('data/model2/usage-extract.xlsx', index=False)
print(f"  Model 1: {len(df_usage[df_usage['equipment_model_id'] == 1])} 条记录")
print(f"  Model 2: {len(df_usage[df_usage['equipment_model_id'] == 2])} 条记录")

print("\n完成！数据已分别保存到 data/model1/ 和 data/model2/ 目录中。")