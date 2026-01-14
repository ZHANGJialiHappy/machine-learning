from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.datasets import load_iris, make_blobs
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib  # 用于保存模型

# 分类 - Random Forest
X, y = load_iris(return_X_y=True)

# 定义要尝试的参数
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3]
}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_
print(f"Random Forest准确率: {best_clf.score(X_test, y_test):.2%}")
print(f"最佳参数: {grid_search.best_params_}")
print(f"交叉验证最佳得分: {grid_search.best_score_:.2%}")

# # 分类 - XGBoost
# xgb = XGBClassifier().fit(X_train, y_train)
# print(f"XGBoost准确率: {xgb.score(X_test, y_test):.2%}")

# # 异常检测 - Isolation Forest
# X_anom, _ = make_blobs(n_samples=300, centers=1)
# iso = IsolationForest().fit(X_anom)
# predictions = iso.predict(X_anom)  # -1=异常, 1=正常
# anomalies = (predictions == -1).sum()
# print(f"检测到 {anomalies} 个异常点 (共{len(X_anom)}个)")

# 保存模型
joblib.dump(best_clf, 'models/random_forest_model.pkl')
# joblib.dump(xgb, 'xgboost_model.pkl')
# joblib.dump(iso, 'isolation_forest_model.pkl')
# print("\n✓ 模型已保存到文件")