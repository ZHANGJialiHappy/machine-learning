import csv
from collections import Counter

def read_customer_ids(filename):
    """读取CSV文件中的customer_id"""
    customer_ids = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            customer_ids.append(int(row['customer_id']))
    return customer_ids

# 读取两个CSV文件
print("正在读取文件...")
raw_customer_ids = read_customer_ids('raw.csv')
processed_customer_ids = read_customer_ids('processed.csv')

# 获取唯一的customer_id
raw_unique = set(raw_customer_ids)
processed_unique = set(processed_customer_ids)

# 统计每个customer_id出现的次数
raw_counts = Counter(raw_customer_ids)
processed_counts = Counter(processed_customer_ids)

# 统计信息
print("=" * 60)
print("Customer ID 比较统计")
print("=" * 60)
print(f"\nraw.csv 中的唯一 customer_id 数量: {len(raw_unique)}")
print(f"processed.csv 中的唯一 customer_id 数量: {len(processed_unique)}")
print(f"\nraw.csv 中的总行数: {len(raw_customer_ids)}")
print(f"processed.csv 中的总行数: {len(processed_customer_ids)}")

# 比较customer_id是否一样
if raw_unique == processed_unique:
    print("\n✓ 两个文件中的 customer_id 完全相同")
    print(f"  共有 {len(raw_unique)} 个唯一的 customer_id")
    print(f"  customer_id 范围: {min(raw_unique)} 到 {max(raw_unique)}")
else:
    print("\n✗ 两个文件中的 customer_id 不相同")
    
    # 找出只在raw.csv中存在的customer_id
    only_in_raw = raw_unique - processed_unique
    if only_in_raw:
        print(f"\n  只在 raw.csv 中存在的 customer_id ({len(only_in_raw)} 个):")
        print(f"  {sorted(only_in_raw)}")
    
    # 找出只在processed.csv中存在的customer_id
    only_in_processed = processed_unique - raw_unique
    if only_in_processed:
        print(f"\n  只在 processed.csv 中存在的 customer_id ({len(only_in_processed)} 个):")
        print(f"  {sorted(only_in_processed)}")
    
    # 找出两个文件共有的customer_id
    common_ids = raw_unique & processed_unique
    print(f"\n  两个文件共有的 customer_id: {len(common_ids)} 个")

print("\n" + "=" * 60)

# 详细的customer_id行数统计
print("\n每个 customer_id 的行数统计:")
print("-" * 60)
print(f"{'customer_id':<15} {'raw.csv行数':<15} {'processed.csv行数':<20} {'差异':<10}")
print("-" * 60)

all_ids = sorted(raw_unique | processed_unique)
for cid in all_ids:
    raw_count = raw_counts.get(cid, 0)
    processed_count = processed_counts.get(cid, 0)
    diff = processed_count - raw_count
    print(f"{cid:<15} {raw_count:<15} {processed_count:<20} {diff:<10}")

print("\n" + "=" * 60)
