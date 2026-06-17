import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_excel("original.xlsx")

# ❗只选连续变量（需要标准化的）
zscore_cols = [
    'C_G','C_WBC','C_RBC','C_P','C_N',
    'B_G','B_CRP','B_WBC','B_N','B_Lym','B_PCT','B_AC'
]

# 初始化
scaler = StandardScaler()

# 只对这些列做Z-score
df[zscore_cols] = scaler.fit_transform(df[zscore_cols])

# 保存
df.to_excel("zscore_data.xlsx", index=False)