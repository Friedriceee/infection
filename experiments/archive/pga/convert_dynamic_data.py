"""
将"曲线阳"+"曲线阴"两个 Excel 转换为 dactformer_v21.py 可直接读取的标准动态 CSV.

输出格式 (宽表):
  patient_id, C_G_t0, C_G_t1, C_G_t2, C_G_t3,
              C_WBC_t0, ..., C_P_t3, C_RBC_t0, ..., C_N_t3
  每行一个有动态跟踪的病人

注意事项:
  1. 原始文件中 'WBC_1' 实际是 CSF 白细胞 (C_WBC), 在此显式重命名
  2. transparency 是分类变量, 不送入 TTE 趋势统计 (移除)
  3. 极端值 (如 C_RBC=392000, C_P=3000) 不在此处处理, 留给训练管道做 log1p
  4. 时间点编号: 原文件 _1/_2/_3/_4  →  统一为 _t0/_t1/_t2/_t3
  5. T=1 缺失率极高 (阳性 60%, 阴性 83%), 缺失值保留为 NaN, has_dyn 仍标 1
     (代码加载时会用 0 填充, 但保留 has_dyn=1 让模型仍能从 T=2/T=3/T=4 学)
  6. 在静态数据中找不到的 ID, 这里仍然写入 CSV — 由训练管道 ID 对齐时
     自动排除 (~13 阳性 + 31 阴性)

运行:
  python convert_dynamic_data.py
  → 生成 dynamic_curves.csv
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 输入路径 (按你的实际位置调整)
POS_PATH    = "曲线阳.xlsx"
NEG_PATH    = "曲线阴.xlsx"
STATIC_PATH = "original.xlsx"
OUTPUT_CSV  = "dynamic_curves.csv"

# 动态数据中的 6 个变量 (注意 transparency 是分类, 排除)
DYN_FEATURES = ["C_WBC", "C_RBC", "C_N", "C_G", "C_P"]   # 5 个连续变量
# 原始列名 → 统一列名映射
COL_RENAME = {
    "WBC":   "C_WBC",   # 关键: 这里是 CSF 白细胞而不是血 WBC
    "C_RBC": "C_RBC",
    "C_N":   "C_N",
    "C_G":   "C_G",
    "C_P":   "C_P",
}
N_TIME_POINTS = 4


def transform_one_file(path: str, source_label: str) -> pd.DataFrame:
    """
    把一个 Excel (阳/阴) 转换为标准宽表 DataFrame.
    返回的 DataFrame 列: [patient_id, C_WBC_t0..t3, C_RBC_t0..t3, ..., C_P_t0..t3, source]
    """
    df = pd.read_excel(path)

    out_rows = []
    for _, row in df.iterrows():
        out = {"patient_id": str(row["ID"])}

        # 5 个连续变量 × 4 时间点
        for raw_name, new_name in COL_RENAME.items():
            for t in range(1, N_TIME_POINTS + 1):
                src_col = f"{raw_name}_{t}"
                dst_col = f"{new_name}_t{t-1}"   # _1 → _t0, _4 → _t3

                if src_col in row.index:
                    val = pd.to_numeric(row[src_col], errors="coerce")
                    out[dst_col] = float(val) if pd.notna(val) else np.nan
                else:
                    out[dst_col] = np.nan

        out["source"] = source_label
        out_rows.append(out)

    return pd.DataFrame(out_rows)


def main():
    # 读取 + 转换
    df_pos = transform_one_file(POS_PATH, "positive")
    df_neg = transform_one_file(NEG_PATH, "negative")

    print(f"阳性曲线: {len(df_pos)} 行")
    print(f"阴性曲线: {len(df_neg)} 行")

    # 合并
    df_all = pd.concat([df_pos, df_neg], ignore_index=True)

    # 重复 ID 检查
    n_dup = df_all["patient_id"].duplicated().sum()
    if n_dup > 0:
        print(f"⚠️ 发现 {n_dup} 个重复 patient_id, 仅保留首次出现")
        df_all = df_all.drop_duplicates(subset="patient_id", keep="first").reset_index(drop=True)

    # ID 类型一致性 (静态 ID 既有数字 22143063 也有字符串 'D39401', 全部转为字符串)
    print(f"\n合并后总行数: {len(df_all)}")

    # 与静态数据比对 ID 对齐情况
    if Path(STATIC_PATH).exists():
        static_df = pd.read_excel(STATIC_PATH)
        static_ids = set(static_df["ID"].astype(str).tolist())
        dyn_ids = set(df_all["patient_id"].tolist())

        matched = dyn_ids & static_ids
        unmatched = dyn_ids - static_ids
        print(f"\n=== ID 对齐验证 ===")
        print(f"动态 unique ID: {len(dyn_ids)}")
        print(f"静态 unique ID: {len(static_ids)}")
        print(f"成功对齐: {len(matched)} (将参与训练)")
        print(f"动态有但静态无: {len(unmatched)} (将被自动排除)")

    # 缺失率快报
    print(f"\n=== 各时间点缺失率 ===")
    for t in range(N_TIME_POINTS):
        cols_t = [c for c in df_all.columns if c.endswith(f"_t{t}")]
        miss_avg = df_all[cols_t].isnull().mean().mean()
        print(f"T{t}: 平均缺失 {miss_avg*100:.1f}%  ({cols_t})")

    # 标签一致性验证 (阳性曲线对应 outcome=1, 阴性曲线对应 outcome=0)
    if Path(STATIC_PATH).exists():
        static_df["ID_str"] = static_df["ID"].astype(str)
        merged = df_all.merge(
            static_df[["ID_str", "outcome"]],
            left_on="patient_id", right_on="ID_str", how="left"
        )
        # 阳性源 → outcome 应该都是 1
        pos_sub = merged[merged["source"] == "positive"]["outcome"].dropna()
        neg_sub = merged[merged["source"] == "negative"]["outcome"].dropna()
        print(f"\n=== 标签一致性 ===")
        print(f"阳性曲线 → outcome=1 比例: {(pos_sub == 1).mean():.3f} (n={len(pos_sub)})")
        print(f"阴性曲线 → outcome=0 比例: {(neg_sub == 0).mean():.3f} (n={len(neg_sub)})")

    # 保存 (去掉 source 列, 因为模型不用它, label 从静态来)
    df_to_save = df_all.drop(columns=["source"])
    df_to_save.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ 已保存: {OUTPUT_CSV}  ({len(df_to_save)} 行 × {df_to_save.shape[1]} 列)")
    print(f"\n样本前 3 行:")
    print(df_to_save.head(3).to_string())

if __name__ == "__main__":
    main()
