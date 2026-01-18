#!/bin/bash

# Prophet 模型训练和预测 - 完整流程示例
# 
# 使用方法:
#   chmod +x run_example.sh
#   ./run_example.sh

echo "=========================================="
echo "Prophet 咖啡豆消耗预测 - 完整流程演示"
echo "=========================================="

# 检查虚拟环境
if [ -d "venv" ]; then
    echo ""
    echo "检测到虚拟环境，请先激活:"
    echo "  source venv/bin/activate"
    echo ""
    read -p "按 Enter 继续（确保已激活虚拟环境）..."
fi

# 步骤1: 训练模型
echo ""
echo "=========================================="
echo "步骤 1/3: 训练 Prophet 模型"
echo "=========================================="
python train_prophet.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 训练失败！请检查:"
    echo "  1. 是否已安装依赖: pip install prophet pandas numpy"
    echo "  2. 数据文件是否存在: data/train_data/daily_consumption.csv"
    exit 1
fi

# 步骤2: 使用模型预测
echo ""
echo "=========================================="
echo "步骤 2/3: 使用训练好的模型进行预测"
echo "=========================================="
python predict.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 预测失败！"
    exit 1
fi

# 步骤3: 可选 - 模型验证
echo ""
echo "=========================================="
echo "步骤 3/3: 模型性能验证（可选）"
echo "=========================================="
read -p "是否运行交叉验证？(y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python validate_model.py
else
    echo "跳过验证步骤"
fi

# 完成
echo ""
echo "=========================================="
echo "✓ 完整流程演示完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - models/prophet_model.json          (训练好的模型)"
echo "  - predictions/forecast_30days.csv    (训练时的预测结果)"
echo "  - predictions/latest_forecast.csv    (最新预测结果)"
echo ""
echo "下一步:"
echo "  1. 查看预测结果: cat predictions/latest_forecast.csv"
echo "  2. 在生产环境中使用 predict.py 进行实时预测"
echo "  3. 定期更新训练数据并重新训练模型"
echo ""
