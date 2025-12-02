
## 🎯 模型性能

- **准确率**: 80.3%
- **AUC**: 0.824
- **特异度**: 92.3% (优秀的无感染识别能力)
- **敏感度**: 50.9% (感染识别能力)

## 📁 项目结构

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

## 🚀 快速开始


### 方法一：推荐方式（使用虚拟环境）

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行数据转换
python data_transformer.py

# 5. 训练AMFormer模型
python train_amformer.py

# 6. 分析特征重要性 (可选)
python feature_importance_analysis.py
```

### 方法二：直接安装（如果不想用虚拟环境）

```bash
# 直接安装依赖包
pip install pandas numpy scikit-learn torch openpyxl

# 按顺序运行程序
python data_transformer.py
python train_amformer.py
```

## 📊 数据说明

### 输入数据
- **original.xlsx**: 915条临床记录，包含26个原始特征
- **feature.xlsx**: 20个特征的医学分级标准和临床意义

### 特征类型
- **脑脊液指标**: C_G(葡萄糖), C_WBC(白细胞), C_RBC(红细胞), C_P(蛋白), C_N(中性粒细胞%), transparency(透明度)
- **临床评估**: GCS(格拉斯哥昏迷评分), tem(体温), sex(性别)
- **治疗相关**: tube(引流管), site(手术部位), other_inf(其他感染)
- **血液指标**: B_G(血糖), B_CRP(CRP), B_WBC(白细胞), B_N(中性粒细胞%), B_Lym(淋巴细胞%), B_PCT(PCT), B_AC(乳酸)

### 输出结果
- **目标变量**: outcome (0=无感染, 1=感染)
- **编码数据**: 所有特征按医学标准转换为0-3的分级编码

## 🎯 特征重要性排名

基于随机森林算法分析的TOP 5重要特征：

1. **C_G (脑脊液葡萄糖)**: 18.4% - 感染诊断的金标准
2. **B_CRP (血CRP)**: 6.6% - 炎症反应标志物
3. **tem (体温)**: 6.5% - 感染的基础指标
4. **GCS (昏迷评分)**: 5.9% - 神经功能评估
5. **B_PCT (血PCT)**: 5.9% - 细菌感染标志物

## 🔧 常见问题

### Q: 需要发送venv虚拟环境文件夹吗？
**A: 不需要！** 只需要发送以下文件：
- 所有 `.py` 文件
- `requirements.txt`
- `original.xlsx` 和 `feature.xlsx`
- `README.md`

接收方按照README说明重新创建虚拟环境即可。

### Q: 如果遇到依赖包安装问题怎么办？
```bash
# 升级pip
pip install --upgrade pip

# 如果torch安装慢，使用国内镜像
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 或者分别安装
pip install pandas numpy scikit-learn openpyxl
pip install torch
```

### Q: 程序运行出错怎么办？
1. **文件找不到**: 确保 `original.xlsx` 和 `feature.xlsx` 在同一目录
2. **内存不足**: 可以在 `train_amformer.py` 中将 `batch_size=64` 改为 `batch_size=32`
3. **CUDA错误**: 程序会自动使用CPU，无需GPU

### Q: 如何修改模型参数？
在 `train_amformer.py` 中可以调整：
- `EPOCHS = 50` (训练轮数)
- `batch_size=64` (批次大小)
- `embed_dim=32` (嵌入维度)
- `lr=1e-3` (学习率)

## 📈 输出文件说明

运行完成后会生成：
- `转化后_编码数据_最终版本.csv`: AMFormer训练数据 (915×21)




