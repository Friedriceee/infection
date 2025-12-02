import pandas as pd
import numpy as np
import re

def load_and_clean_original_data():
    """
    加载和清理原始数据
    """
    # 读取原始Excel文件，跳过前两行
    df = pd.read_excel('original.xlsx', skiprows=2)
    
    # 手动映射列名（基于观察到的数据结构）
    column_mapping = {
        'Unnamed: 0': 'outcome',
        '2.4-3.9（最低检测限是1.1，小于1.1按1.1算）': 'C_G',
        '<100': 'C_WBC',
        0: 'C_RBC',
        '120-600（最高检测限是3000，大于3000按3000算）': 'C_P',
        'Unnamed: 5': 'C_N',
        'Unnamed: 6': 'transparency',
        '1-15分（正常15，轻昏迷12-14，中昏迷11-9，深1-8）': 'GCS',
        'Unnamed: 8': 'age',  
        'Unnamed: 9': 'sex',
        'Unnamed: 10': 'tem',
        'Unnamed: 11': 'tube',
        'Unnamed: 12': 'site',
        'Unnamed: 13': 'other_inf',
        'Unnamed: 14': 'B_G',
        '<8.2（最高检测限是200，大于200按200算））': 'B_CRP',
        '3.5-9.5': 'B_WBC',
        '40-75': 'B_N',
        '20-50': 'B_Lym',
        '<0.5': 'B_PCT',
        '0-2': 'B_AC',
        'Unnamed: 21': 'unused1',  # 空列
        'Unnamed: 22': 'ID',  # 修正：第22列是真正的ID
        'Unnamed: 23': 'date',
        'Unnamed: 24': 'diagnosis',
        'Unnamed: 25': 'B_RBC'
    }
    
    # 重命名列
    df = df.rename(columns=column_mapping)
    
    # 移除完全为空的行
    df = df.dropna(how='all')
    
    # 移除第一行（包含列名信息的行）
    if len(df) > 0 and df.iloc[0]['outcome'] == 'outcome':
        df = df.iloc[1:].reset_index(drop=True)
    
    print(f"清理后数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print("前5行数据:")
    print(df.head())
    
    return df

def load_feature_rules():
    """
    加载特征分级规则
    """
    df_feature = pd.read_excel('feature.xlsx')
    
    # 创建特征规则字典
    feature_rules = {}
    current_feature = None
    
    for _, row in df_feature.iterrows():
        if pd.notna(row['特征名']):
            current_feature = row['特征名']
            feature_rules[current_feature] = []
        
        if current_feature and pd.notna(row['分段区间']):
            rule = {
                'range': row['分段区间'],
                'code': row['分段编码'],
                'description': row['分段名称（临床意义）']
            }
            feature_rules[current_feature].append(rule)
    
    print("特征规则加载完成:")
    for feature, rules in feature_rules.items():
        print(f"{feature}: {len(rules)} 个分级")
    
    return feature_rules

def parse_range_condition(range_str):
    """
    解析范围条件字符串，返回判断函数
    """
    range_str = str(range_str).strip()
    
    def safe_convert_to_number(x):
        """安全地将值转换为数字"""
        if pd.isna(x):
            return None
        try:
            # 尝试转换为浮点数
            return float(x)
        except (ValueError, TypeError):
            return None
    
    # 处理特殊情况
    if range_str == '0':
        return lambda x: safe_convert_to_number(x) == 0
    elif range_str == '>0':
        def check_gt_zero(x):
            num = safe_convert_to_number(x)
            return num is not None and num > 0
        return check_gt_zero
    elif range_str == '无':
        return lambda x: pd.isna(x) or x == 0 or x == '无' or str(x).strip() == ''
    elif range_str == '有':
        return lambda x: pd.notna(x) and x != 0 and x != '无' and str(x).strip() != ''
    
    # 处理数值范围
    if '<' in range_str and '–' not in range_str and '-' not in range_str:
        # 小于条件 如 <2.4, <100
        match = re.search(r'<(\d+\.?\d*)', range_str)
        if match:
            threshold = float(match.group(1))
            def check_lt(x):
                num = safe_convert_to_number(x)
                return num is not None and num < threshold
            return check_lt
    
    elif '>' in range_str and '–' not in range_str and '-' not in range_str:
        # 大于条件 如 >4.5, >1000
        match = re.search(r'>(\d+\.?\d*)', range_str)
        if match:
            threshold = float(match.group(1))
            def check_gt(x):
                num = safe_convert_to_number(x)
                return num is not None and num > threshold
            return check_gt
    
    elif '–' in range_str or '-' in range_str:
        # 范围条件 如 2.4–4.5, 100–1000
        # 统一使用–作为分隔符
        range_str = range_str.replace('-', '–')
        parts = range_str.split('–')
        if len(parts) == 2:
            try:
                min_val = float(parts[0])
                max_val = float(parts[1])
                def check_range(x):
                    num = safe_convert_to_number(x)
                    return num is not None and min_val <= num <= max_val
                return check_range
            except ValueError:
                pass
    
    elif range_str.isdigit():
        # 精确匹配 如 15, 1, 2
        target = int(range_str)
        def check_exact(x):
            num = safe_convert_to_number(x)
            return num is not None and int(num) == target
        return check_exact
    
    # 默认返回False的函数
    return lambda x: False

def apply_feature_encoding(df, feature_rules):
    """
    应用特征编码规则
    """
    encoded_df = df.copy()
    
    for feature_name, rules in feature_rules.items():
        if feature_name not in df.columns:
            print(f"警告: 特征 {feature_name} 不在数据中")
            continue
            
        # 创建编码列
        encoded_col = f"{feature_name}_encoded"
        encoded_df[encoded_col] = np.nan
        
        print(f"\n处理特征: {feature_name}")
        
        for rule in rules:
            range_condition = parse_range_condition(rule['range'])
            code = rule['code']
            
            # 应用条件
            mask = df[feature_name].apply(range_condition)
            encoded_df.loc[mask, encoded_col] = code
            
            print(f"  范围 {rule['range']} -> 编码 {code}: {mask.sum()} 条记录")
    
    return encoded_df

def main():
    """
    主函数：执行数据转换流程
    """
    print("=== 开始数据转换流程 ===")
    
    # 1. 加载和清理原始数据
    print("\n1. 加载和清理原始数据")
    df_original = load_and_clean_original_data()
    
    # 2. 加载特征规则
    print("\n2. 加载特征分级规则")
    feature_rules = load_feature_rules()
    
    # 3. 应用编码规则
    print("\n3. 应用特征编码")
    df_encoded = apply_feature_encoding(df_original, feature_rules)
    
    # 4. 生成AMFormer可直接使用的数据
    print("\n4. 生成AMFormer训练数据")
    encoded_columns = [col for col in df_encoded.columns if col.endswith('_encoded')]
    amformer_columns = ['ID'] + encoded_columns
    
    # 创建AMFormer数据集
    amformer_df = df_encoded[amformer_columns].copy()
    
    # 重命名编码列（去掉_encoded后缀）
    rename_dict = {col: col.replace('_encoded', '') for col in encoded_columns}
    amformer_df = amformer_df.rename(columns=rename_dict)
    
    # 保存AMFormer数据
    amformer_file = "转化后_编码数据_最终版本.csv"
    amformer_df.to_csv(amformer_file, index=False, encoding='utf-8-sig')
   
    # 5. 显示转换统计
    print("\n5. 转换统计:")
    for col in encoded_columns:
        print(f"{col}: {df_encoded[col].value_counts().to_dict()}")
    
    return df_encoded

if __name__ == "__main__":
    result = main()