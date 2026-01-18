import pandas as pd
import numpy as np

df = pd.read_csv("data/mock_data/part-00000-f594e922-39f9-49c9-b415-3767311633ef-c000.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 用 mean 作为点预测
df["pred"] = df["mean"]
df["actual"] = df["target_value"]

def wape(x):
    return np.abs(x["pred"] - x["actual"]).sum() / x["actual"].sum()

# === 1) 每台机器 WAPE ===
wape_per_item = df.groupby("item").apply(wape)
print("\nWAPE per item:")
print(wape_per_item)

print("\nOverall WAPE:", wape(df))

# === 2) 缺豆风险 ===
df["under_predict"] = df["pred"] < df["actual"]
df["under_90"] = df["pred"] < df["actual"] * 0.9

risk = df.groupby("item")[["under_predict", "under_90"]].mean()
print("\nUnder-prediction risk:")
print(risk)

# === 3) Baseline: Naive（昨天 = 今天）===
df = df.sort_values(["item", "timestamp"])
df["naive"] = df.groupby("item")["actual"].shift(1)

baseline = df.dropna().copy()
baseline["pred"] = baseline["naive"]

print("\nNaive baseline WAPE:", wape(baseline))
