# 中枢神经系统感染风险预测临床辅助决策系统

Clinical Decision Support System for CNS Infection Risk Prediction

## 项目简介

这是一个用于毕业设计展示的临床辅助决策原型系统，使用 Flask + SQLite + Jinja2 + HTML/CSS/JavaScript + ECharts 构建。系统支持医生登录、患者管理、多时间点病例记录、静态风险预测、动态风险预测、历史预测保存、趋势图展示、辅助决策报告和基础日志审计。

## 功能说明

- 医生和管理员登录，密码使用哈希存储。
- 新增、查询和查看患者详情。
- 支持 Excel / CSV 批量导入患者 ID 和既往检测数据。
- 同一患者支持录入多条病例记录。
- 新增病例后自动生成静态感染风险预测。
- 同一患者病例数达到 2 条后自动生成动态趋势融合预测。
- 保存所有预测结果，并更新患者当前风险状态。
- 患者详情页展示病例时间轴、历史预测、风险趋势图和核心指标趋势图。
- 报告页展示患者摘要、预测结果、趋势图、关键影响指标、临床提示和模型版本，支持浏览器打印保存 PDF。
- 管理员可查看操作日志，并可创建医生或管理员账号。

## 项目结构

```text
cns_prediction_system/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── database.db
├── models/
├── services/
├── static/
├── templates/
└── utils/
```

`database.db` 会在首次启动时自动创建并初始化。

## 安装依赖

```bash
cd cns_prediction_system
pip install -r requirements.txt
```

## 启动方式

```bash
python app.py
```

默认访问：

```text
http://127.0.0.1:5000
```

## 默认账号

- 医生账号：`doctor / 123456`
- 管理员账号：`admin / 123456`

## 页面说明

- `/login`：登录页。
- `/`：首页。
- `/patients`：患者列表和查询。
- `/patients/new`：新增患者。
- `/patients/import`：批量导入患者与检测数据，支持 `original.xlsx` 静态基础数据表和 `曲线阳.csv / 曲线阴.csv` 动态曲线训练表结构。
- `/patients/<patient_id>`：患者详情、病例时间轴、预测记录和趋势图。
- `/cases/new?patient_id=<patient_id>`：新增病例，保存后自动预测。
- `/cases/<case_id>`：病例详情。
- `/predictions/<prediction_id>`：预测详情。
- `/report/<patient_id>`：辅助决策报告。
- `/model_versions`：模型版本展示。
- `/logs`：管理员操作日志。
- `/admin/users`：管理员账号管理。
- `/admin/users/new`：管理员新增账号。

## 模型替换说明

当前真实模型推理适配层位于：

```text
utils/real_model.py
```

系统会加载项目上级 `web_model/` 目录中的模型文件：

```text
web_model/pga2.py
web_model/best_fold1.pth
web_model/static_scaler_fold1.joblib
web_model/dynamic_full_model_fold1.pth
web_model/dynamic_head_fold1.pth
web_model/dyn_scaler_fold1.joblib
```

如需再次替换模型，保持以下函数输入输出格式即可：

```python
def predict_static(case_data: dict) -> dict:
    ...

def predict_dynamic(case_list: list, latest_static_score: float) -> dict:
    ...
```

静态模型默认名称为 `PGFormer v1.0`，动态模型默认名称为 `D-PGFormer v1.0`。风险等级规则位于 `utils/risk_level.py`。

## 批量导入说明

导入入口位于 `/patients/import`。当前支持：

- `.xlsx` 静态基础数据表：识别 `ID`、`time`、`age`、`sex`、`tem`、`tube`、`site`、`other_inf`、`B_*`、`C_*` 等字段。
- `.csv` 动态曲线表：识别 `ID`、`WBC_1`、`C_RBC_1`、`C_N_1`、`transparency_1`、`C_G_1`、`C_P_1` 到 `_4` 的多时间点字段。

导入规则：

- `ID` 会作为患者编号；如果患者已存在则复用，否则自动创建。
- 静态表中一行导入为一条病例。
- 动态曲线表中一行会展开为同一患者的多条时间点病例。
- 动态曲线表如果没有 `D1-D4` 字段，会使用 `time` 作为首个检查时间，并默认每个时间点相隔 1 天。
- 导入患者的科室统一设为 `神经内科`。
- 导入病例后自动执行静态预测；同一患者病例数达到 2 条后自动执行动态预测。
- 性别字段按 `0=男、1=女` 映射，缺失或其他值显示为 `未知`。
- `tube`、`other_inf` 的 `1/0` 会映射为 `是/否`，`site` 的 `1/0` 会映射为 `枕叶/非枕叶`。
- 输入指标不完整时不会阻断预测，缺失数值会按模型适配层的默认值处理，以保证导入和预测流程可运行。

## 注意事项

- 本系统仅用于原型演示和毕业设计答辩，不可直接用于真实临床诊疗。
- 辅助决策声明：本系统结果仅作为临床辅助参考，不能替代医生的临床诊断和治疗决策。
- Flask 启动时已关闭 debug 模式。
- ECharts 通过 CDN 引入，离线环境下可将 ECharts 文件下载后改为本地静态引用。
