# 数据库与数据表说明

本系统使用 SQLite 作为本地关系型数据库，数据库文件位于项目根目录 `database.db`。系统启动时会自动初始化数据表、默认账号和模型版本。

## users

医生和管理员账号表。密码使用 Werkzeug 哈希存储。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | INTEGER PRIMARY KEY AUTOINCREMENT | 用户主键 |
| username | TEXT UNIQUE NOT NULL | 登录用户名 |
| password_hash | TEXT NOT NULL | 密码哈希 |
| role | TEXT NOT NULL | 用户角色：doctor/admin |
| real_name | TEXT | 真实姓名 |
| department | TEXT | 科室 |
| created_at | TEXT | 创建时间 |
| last_login_at | TEXT | 最后登录时间 |
| is_active | INTEGER DEFAULT 1 | 是否启用 |

## patients

患者基本信息表。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | INTEGER PRIMARY KEY AUTOINCREMENT | 患者主键 |
| patient_no | TEXT UNIQUE NOT NULL | 患者编号 |
| name | TEXT NOT NULL | 姓名 |
| sex | TEXT NOT NULL | 性别 |
| age | INTEGER NOT NULL | 年龄 |
| department | TEXT | 科室 |
| inpatient_no | TEXT | 住院号 |
| outpatient_no | TEXT | 门诊号 |
| admission_time | TEXT | 入院时间 |
| remark | TEXT | 备注 |
| current_risk_score | REAL | 当前风险概率 |
| current_risk_level | TEXT | 当前风险等级 |
| created_at | TEXT | 创建时间 |
| updated_at | TEXT | 更新时间 |
| is_deleted | INTEGER DEFAULT 0 | 逻辑删除标记 |

## cases

病例检测记录表，同一患者可以有多条检测记录。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | INTEGER PRIMARY KEY AUTOINCREMENT | 病例主键 |
| patient_id | INTEGER NOT NULL | 患者 ID |
| visit_time | TEXT NOT NULL | 检查时间 |
| doctor_id | INTEGER | 录入医生 ID |
| temperature | REAL | 体温 |
| gcs | REAL | GCS 评分 |
| tube | TEXT | 是否置管 |
| site | TEXT | 感染部位：枕叶/非枕叶 |
| other_inf | TEXT | 是否存在其他感染 |
| transparency | TEXT | 脑脊液透明度 |
| B_G | REAL | 血糖 |
| B_WBC | REAL | 血白细胞 |
| B_N | REAL | 血中性粒细胞比例 |
| B_Lym | REAL | 血淋巴细胞比例 |
| B_CRP | REAL | CRP |
| B_PCT | REAL | PCT |
| B_AC | REAL | 乳酸 |
| B_RBC | REAL | 血红细胞 |
| C_G | REAL | 脑脊液葡萄糖 |
| C_WBC | REAL | 脑脊液白细胞 |
| C_RBC | REAL | 脑脊液红细胞 |
| C_P | REAL | 脑脊液蛋白 |
| C_N | REAL | 脑脊液中性粒细胞比例 |
| remark | TEXT | 备注 |
| created_at | TEXT | 创建时间 |
| updated_at | TEXT | 更新时间 |
| is_deleted | INTEGER DEFAULT 0 | 逻辑删除标记 |

## predictions

预测结果表，保存静态预测和动态预测历史。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | INTEGER PRIMARY KEY AUTOINCREMENT | 预测主键 |
| patient_id | INTEGER NOT NULL | 患者 ID |
| case_id | INTEGER | 关联病例 ID |
| prediction_type | TEXT NOT NULL | static/dynamic |
| model_name | TEXT NOT NULL | 模型名称 |
| model_version | TEXT NOT NULL | 模型版本 |
| risk_score | REAL NOT NULL | 风险概率 |
| risk_level | TEXT NOT NULL | 风险等级 |
| static_score | REAL | 静态风险 |
| dynamic_score | REAL | 动态融合风险 |
| dynamic_delta | REAL | 动态修正值 |
| input_snapshot | TEXT | 输入快照 JSON |
| result_json | TEXT | 预测结果 JSON |
| status | TEXT DEFAULT 'valid' | 记录状态 |
| created_by | INTEGER | 创建用户 ID |
| created_at | TEXT | 创建时间 |

## operation_logs

操作日志表，用于记录关键操作。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | INTEGER PRIMARY KEY AUTOINCREMENT | 日志主键 |
| user_id | INTEGER | 操作用户 ID |
| action | TEXT NOT NULL | 操作类型 |
| target_type | TEXT | 操作对象类型 |
| target_id | INTEGER | 操作对象 ID |
| description | TEXT | 操作描述 |
| ip_address | TEXT | IP 地址 |
| created_at | TEXT | 操作时间 |

## model_versions

模型版本表。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | INTEGER PRIMARY KEY AUTOINCREMENT | 模型版本主键 |
| model_name | TEXT NOT NULL | 模型名称 |
| model_version | TEXT NOT NULL | 版本号 |
| model_type | TEXT NOT NULL | static/dynamic |
| description | TEXT | 模型说明 |
| input_features | TEXT | 输入特征说明 |
| status | TEXT DEFAULT 'enabled' | 状态 |
| created_at | TEXT | 创建时间 |
| updated_at | TEXT | 更新时间 |

