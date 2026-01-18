import csv, random
from datetime import date, timedelta

random.seed(7)

start = date(2025, 7, 1)
days = 90
items = {
    "M004": 2.2,
}

def weekly_factor(d):
    # 周末略高一点，模拟真实波动
    return 1.15 if d.weekday() >= 5 else 1.0

rows = []
for i in range(days):
    d = start + timedelta(days=i)
    for item, base in items.items():
        # 少量缺失/停用日，用 0
        if random.random() < 0.05:
            y = 0.0
        else:
            noise = random.uniform(-0.15, 0.15)  # 轻微噪声
            y = max(0.0, round(base * weekly_factor(d) * (1 + noise), 2))
        rows.append((d.isoformat(), item, y))

with open("data/mock_data/beans_daily_usage_item04.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "item_id", "target"])
    w.writerows(rows)

print("wrote beans_daily_usage_90d.csv with", len(rows), "rows")

