
## 模型性能

- **准确率**: 80.3%
- **AUC**: 0.824
- **特异度**: 92.3% (优秀的无感染识别能力)
- **敏感度**: 50.9% (感染识别能力)

## 运行步骤
- 运行 data_transformer.py 先转化原始数据表
- 运行 train_amformer.py

## 项目结构

```
├── original.xlsx                    # 原始临床数据 (915条记录)
├── feature.xlsx                     # 特征分级规则 (20个特征)
├── data_transformer.py              # 数据转换程序 (一步到位)
├── train_amformer.py                # AMFormer模型训练程序
├── feature_importance_analysis.py   # 特征重要性分析
├── requirements.txt                 # Python依赖包清单
├── README.md                       # 项目说明文档
└── 转化后_编码数据_最终版本.csv      # 最终编码数据 (915×21)
```






