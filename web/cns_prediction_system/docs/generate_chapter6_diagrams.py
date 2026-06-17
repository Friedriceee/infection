from pathlib import Path
from math import atan2, cos, sin, pi
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parent / "chapter6_diagrams"


COLORS = {
    "blue": "#dbeafe",
    "blue2": "#60a5fa",
    "navy": "#1d4ed8",
    "orange": "#fed7aa",
    "green": "#dcfce7",
    "green2": "#86efac",
    "gray": "#f3f4f6",
    "gray2": "#e5e7eb",
    "purple": "#e0e7ff",
    "purple2": "#a5b4fc",
    "red": "#fee2e2",
    "white": "#ffffff",
    "line": "#374151",
    "muted": "#6b7280",
}


def puml_header(title: str) -> str:
    return f"""@startuml
title {title}
skinparam backgroundColor #FFFFFF
skinparam defaultFontName "PingFang SC"
skinparam shadowing false
skinparam roundcorner 10
skinparam ArrowColor #374151
skinparam RectangleBorderColor #374151
skinparam RectangleBackgroundColor #F8FAFC
skinparam PackageBorderColor #64748B
skinparam PackageBackgroundColor #F8FAFC
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class Svg:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="{width/2}" y="54" text-anchor="middle" font-family="PingFang SC, Microsoft YaHei, Arial" font-size="34" font-weight="700" fill="#111827">{escape(title)}</text>',
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#374151"/></marker></defs>',
        ]

    def rect(self, x, y, w, h, fill=COLORS["white"], stroke=COLORS["line"], r=12, sw=2, dashed=False):
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>'
        )

    def text(self, x, y, text, size=24, weight=400, color="#111827", anchor="middle"):
        lines = str(text).split("\n")
        for i, line in enumerate(lines):
            yy = y + i * (size + 8)
            self.items.append(
                f'<text x="{x}" y="{yy}" text-anchor="{anchor}" font-family="PingFang SC, Microsoft YaHei, Arial" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(line)}</text>'
            )

    def box(self, x, y, w, h, text, fill=COLORS["gray"], stroke=COLORS["line"], size=23, weight=500, r=8):
        self.rect(x, y, w, h, fill, stroke, r)
        self.text(x + w / 2, y + h / 2 + 8 - (text.count("\n") * 13), text, size=size, weight=weight)

    def line(self, x1, y1, x2, y2, arrow=False, dashed=False, sw=3, color=COLORS["line"]):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        self.items.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{dash}{marker}/>'
        )

    def arrow(self, x1, y1, x2, y2, dashed=False, label=None):
        self.line(x1, y1, x2, y2, arrow=True, dashed=dashed)
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 - 8, label, size=18, color=COLORS["muted"])

    def polyline(self, points, arrow=False, dashed=False, label=None):
        pts = " ".join(f"{x},{y}" for x, y in points)
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        self.items.append(f'<polyline points="{pts}" fill="none" stroke="#374151" stroke-width="3"{dash}{marker}/>')
        if label and len(points) >= 2:
            x1, y1 = points[len(points)//2 - 1]
            x2, y2 = points[len(points)//2]
            self.text((x1+x2)/2, (y1+y2)/2 - 8, label, size=18, color=COLORS["muted"])

    def save(self, path: Path):
        self.items.append("</svg>")
        write(path, "\n".join(self.items))


def _parse_float(value, default=0.0):
    try:
        return float(str(value).replace("px", ""))
    except Exception:
        return default


def _font(size, weight=400):
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc" if int(weight or 400) >= 600 else "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _draw_arrowhead(draw, start, end, color, scale):
    x1, y1 = start
    x2, y2 = end
    angle = atan2(y2 - y1, x2 - x1)
    length = 16 * scale
    spread = pi / 7
    p1 = (x2 - length * cos(angle - spread), y2 - length * sin(angle - spread))
    p2 = (x2 - length * cos(angle + spread), y2 - length * sin(angle + spread))
    draw.polygon([(x2, y2), p1, p2], fill=color)


def render_svg_subset_to_png(svg_path: Path, png_path: Path, scale: float = 1.6) -> None:
    """Render the small SVG subset emitted by this script to PNG with real Chinese text."""
    from PIL import Image, ImageDraw

    root = ET.parse(svg_path).getroot()
    width = int(_parse_float(root.attrib.get("width"))) 
    height = int(_parse_float(root.attrib.get("height")))
    image = Image.new("RGB", (int(width * scale), int(height * scale)), "white")
    draw = ImageDraw.Draw(image)

    def sx(v):
        return _parse_float(v) * scale

    def draw_dashed_line(points, fill, width, dash=14):
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            dx, dy = x2 - x1, y2 - y1
            length = max((dx * dx + dy * dy) ** 0.5, 1)
            ux, uy = dx / length, dy / length
            pos = 0
            while pos < length:
                end = min(pos + dash * scale, length)
                draw.line((x1 + ux * pos, y1 + uy * pos, x1 + ux * end, y1 + uy * end), fill=fill, width=width)
                pos += dash * scale * 1.8

    def walk(node):
        tag = node.tag.split("}")[-1]
        if tag == "defs":
            return
        if tag == "rect":
            x, y, w, h = sx(node.attrib.get("x", 0)), sx(node.attrib.get("y", 0)), sx(node.attrib.get("width", 0)), sx(node.attrib.get("height", 0))
            fill = node.attrib.get("fill", "white")
            outline = None if node.attrib.get("stroke") == "none" else node.attrib.get("stroke", "#000000")
            sw = max(1, int(_parse_float(node.attrib.get("stroke-width", 1)) * scale))
            radius = sx(node.attrib.get("rx", 0))
            draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=sw)
        elif tag == "line":
            p1 = (sx(node.attrib.get("x1", 0)), sx(node.attrib.get("y1", 0)))
            p2 = (sx(node.attrib.get("x2", 0)), sx(node.attrib.get("y2", 0)))
            fill = node.attrib.get("stroke", "#000000")
            sw = max(1, int(_parse_float(node.attrib.get("stroke-width", 1)) * scale))
            if "stroke-dasharray" in node.attrib:
                draw_dashed_line([p1, p2], fill, sw)
            else:
                draw.line((*p1, *p2), fill=fill, width=sw)
            if "marker-end" in node.attrib:
                _draw_arrowhead(draw, p1, p2, fill, scale)
        elif tag == "polyline":
            raw = node.attrib.get("points", "").strip().split()
            pts = [(float(p.split(",")[0]) * scale, float(p.split(",")[1]) * scale) for p in raw if "," in p]
            fill = node.attrib.get("stroke", "#000000")
            sw = max(1, int(_parse_float(node.attrib.get("stroke-width", 1)) * scale))
            if len(pts) >= 2:
                if "stroke-dasharray" in node.attrib:
                    draw_dashed_line(pts, fill, sw)
                else:
                    draw.line(pts, fill=fill, width=sw, joint="curve")
                if "marker-end" in node.attrib:
                    _draw_arrowhead(draw, pts[-2], pts[-1], fill, scale)
        elif tag == "text":
            text = "".join(node.itertext())
            if text:
                x, y = sx(node.attrib.get("x", 0)), sx(node.attrib.get("y", 0))
                size = max(8, int(_parse_float(node.attrib.get("font-size", 18)) * scale))
                weight = int(_parse_float(node.attrib.get("font-weight", 400), 400))
                fill = node.attrib.get("fill", "#111827")
                font = _font(size, weight)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                anchor = node.attrib.get("text-anchor", "middle")
                tx = x - tw / 2 if anchor == "middle" else x
                draw.text((tx, y - th), text, font=font, fill=fill)
        for child in list(node):
            walk(child)

    for child in list(root):
        walk(child)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)


def render_all_pngs():
    for svg_path in sorted(OUT_DIR.glob("fig6.*.svg")):
        render_svg_subset_to_png(svg_path, OUT_DIR / "png" / svg_path.with_suffix(".png").name)


def fig61():
    puml = puml_header("图6.1 临床辅助决策原型系统功能结构图") + """
rectangle "中枢神经系统感染风险预测\\n临床辅助决策系统" as system {
  package "身份认证与首页" {
    [医生/管理员登录]
    [首页仪表盘]
    [快捷入口]
  }
  package "患者信息管理" {
    [新增患者]
    [查询患者]
    [编辑患者]
    [Excel批量导入]
  }
  package "检测记录管理" {
    [录入检测指标]
    [多时间点病例]
    [动态批量导入]
    [病例详情]
  }
  package "风险预测服务" {
    [静态PGA-AMFormer推理]
    [动态D-PGA-AMFormer推理]
    [缺失值容错预测]
    [风险等级输出]
  }
  package "结果展示与报告" {
    [历史预测记录]
    [风险趋势图]
    [核心指标趋势图]
    [辅助决策报告打印]
  }
  package "系统管理" {
    [账号管理]
    [模型版本展示]
    [操作日志审计]
  }
}
@enduml
"""
    write(OUT_DIR / "fig6.1_system_function_structure.puml", puml)
    s = Svg(1800, 1200, "临床辅助决策原型系统功能结构图")
    s.rect(60, 105, 1680, 1030, "#ffffff", COLORS["line"], 12, 3)
    s.text(900, 165, "中枢神经系统感染风险预测临床辅助决策系统", 32, 700)
    groups = [
        (120, 245, 500, 300, "身份认证与首页", ["医生/管理员登录", "首页仪表盘", "快捷入口"], COLORS["blue"]),
        (650, 245, 500, 300, "患者信息管理", ["新增患者", "查询患者", "编辑患者", "Excel批量导入"], COLORS["green"]),
        (1180, 245, 500, 300, "检测记录管理", ["录入检测指标", "多时间点病例", "动态批量导入", "病例详情"], COLORS["orange"]),
        (120, 625, 500, 350, "风险预测服务", ["静态PGA-AMFormer推理", "动态D-PGA-AMFormer推理", "缺失值容错预测", "风险等级输出"], COLORS["purple"]),
        (650, 625, 500, 350, "结果展示与报告", ["历史预测记录", "风险趋势图", "核心指标趋势图", "辅助决策报告打印"], COLORS["blue"]),
        (1180, 625, 500, 350, "系统管理", ["账号管理", "模型版本展示", "操作日志审计"], COLORS["gray"]),
    ]
    for x, y, w, h, title, items, fill in groups:
        s.rect(x, y, w, h, "#ffffff", COLORS["line"], 10, 3)
        s.text(x+w/2, y+46, title, 30, 700)
        for i, item in enumerate(items):
            bx = x + 52 + (i % 2) * 225
            by = y + 95 + (i // 2) * 110
            s.box(bx, by, 180, 58, item, fill, "#64748b", 22)
    s.save(OUT_DIR / "fig6.1_system_function_structure.svg")


def fig62():
    puml = puml_header("图6.2 系统数据库E-R图") + """
entity users {
  * id : INTEGER <<PK>>
  --
  username : TEXT <<UNIQUE>>
  password_hash : TEXT
  role : TEXT
  real_name : TEXT
  department : TEXT
  last_login_at : TEXT
  is_active : INTEGER
}
entity patients {
  * id : INTEGER <<PK>>
  --
  patient_no : TEXT <<UNIQUE>>
  name : TEXT
  sex : TEXT
  age : INTEGER
  department : TEXT
  admission_time : TEXT
  current_risk_score : REAL
  current_risk_level : TEXT
  is_deleted : INTEGER
}
entity cases {
  * id : INTEGER <<PK>>
  --
  patient_id : INTEGER <<FK>>
  visit_time : TEXT
  doctor_id : INTEGER <<FK>>
  temperature : REAL
  gcs : REAL
  site : TEXT
  other_inf : TEXT
  B_CRP/B_PCT : REAL
  C_WBC/C_P/C_G/C_N : REAL
  is_deleted : INTEGER
}
entity predictions {
  * id : INTEGER <<PK>>
  --
  patient_id : INTEGER <<FK>>
  case_id : INTEGER <<FK>>
  prediction_type : TEXT
  model_name : TEXT
  model_version : TEXT
  risk_score : REAL
  risk_level : TEXT
  static_score : REAL
  dynamic_delta : REAL
  input_snapshot : TEXT
  result_json : TEXT
}
entity operation_logs {
  * id : INTEGER <<PK>>
  --
  user_id : INTEGER <<FK>>
  action : TEXT
  target_type : TEXT
  target_id : INTEGER
  ip_address : TEXT
  created_at : TEXT
}
entity model_versions {
  * id : INTEGER <<PK>>
  --
  model_name : TEXT
  model_version : TEXT
  model_type : TEXT
  description : TEXT
  status : TEXT
  updated_at : TEXT
}
patients ||--o{ cases
patients ||--o{ predictions
cases ||--o{ predictions
users ||--o{ cases
users ||--o{ predictions
users ||--o{ operation_logs
@enduml
"""
    write(OUT_DIR / "fig6.2_database_er.puml", puml)
    s = Svg(1850, 1300, "患者信息与检测记录数据库E-R图")
    entities = [
        (670, 110, 510, 270, "users\n用户表", ["id PK", "username UNIQUE", "password_hash", "role", "real_name / department", "last_login_at / is_active"], COLORS["blue"]),
        (70, 470, 530, 330, "patients\n患者基本信息表", ["id PK", "patient_no UNIQUE", "name / sex / age", "department / admission_time", "current_risk_score", "current_risk_level"], COLORS["green"]),
        (670, 470, 530, 430, "cases\n检测病例记录表", ["id PK", "patient_id FK", "visit_time / doctor_id", "temperature / gcs", "site / other_inf", "B_CRP / B_PCT", "C_WBC / C_P / C_G / C_N"], COLORS["orange"]),
        (1270, 470, 530, 430, "predictions\n预测结果表", ["id PK", "patient_id FK / case_id FK", "prediction_type", "model_name / model_version", "risk_score / risk_level", "static_score / dynamic_delta", "input_snapshot / result_json"], COLORS["purple"]),
        (270, 955, 530, 240, "operation_logs\n操作日志表", ["id PK", "user_id FK", "action / target_type", "target_id / ip_address", "created_at"], COLORS["gray"]),
        (1050, 990, 530, 205, "model_versions\n模型版本表", ["id PK", "model_name / model_version", "model_type / status", "description / updated_at"], COLORS["gray"]),
    ]
    for x, y, w, h, title, attrs, fill in entities:
        s.rect(x, y, w, h, "#ffffff", COLORS["line"], 12, 3)
        s.rect(x, y, w, 70, fill, COLORS["line"], 12, 3)
        s.text(x+w/2, y+32, title, 26, 700)
        for i, a in enumerate(attrs):
            s.text(x+36, y+105+i*32, "• " + a, 21, 500, anchor="start")
    s.arrow(600, 630, 670, 630, label="1:N")
    s.arrow(1200, 630, 1270, 630, label="1:N")
    s.polyline([(335, 470), (335, 335), (670, 240)], arrow=True, label="创建/审计")
    s.polyline([(925, 380), (925, 470)], arrow=True, label="医生录入")
    s.polyline([(925, 900), (535, 955)], arrow=True, label="写日志")
    s.polyline([(1535, 900), (1535, 990)], arrow=True, label="版本对应")
    s.save(OUT_DIR / "fig6.2_database_er.svg")


def fig63():
    puml = puml_header("图6.3 系统总体架构图") + """
rectangle "前端表现层（Jinja2 + HTML/CSS/JavaScript + ECharts）" {
  [登录页] [首页仪表盘] [患者列表/详情] [病例录入/导入] [预测详情/报告] [日志/模型版本]
}
rectangle "请求与页面渲染层（Flask路由）" {
  [登录校验] [表单提交] [文件上传] [页面渲染] [会话管理]
}
rectangle "后端业务服务层（Flask Services）" {
  [auth_service] [patient_service] [case_service] [import_service] [prediction_service] [report_service] [log_service]
}
rectangle "模型推理层（web_model + real_model适配器）" {
  [静态PGA-AMFormer]
  [动态D-PGA-AMFormer]
  [Scaler/Encoder]
  [缺失值容错预处理]
}
database "数据持久层（SQLite）" {
  [users] [patients] [cases] [predictions] [operation_logs] [model_versions]
}
cloud "部署运行层（阿里云ECS）" {
  [Nginx反向代理] [Gunicorn] [Flask应用] [SQLite文件] [web_model模型文件]
}
@enduml
"""
    write(OUT_DIR / "fig6.3_system_architecture.puml", puml)
    s = Svg(1900, 1200, "临床辅助决策原型系统总体架构图")
    layers = [
        ("前端表现层", "Jinja2 模板 / HTML / CSS / JavaScript / ECharts", 120, "#dbeafe", ["登录页", "首页仪表盘", "患者列表/详情", "病例录入/导入", "预测详情/报告", "日志/模型版本"]),
        ("请求与页面渲染层", "Flask 路由、Session 登录态、表单提交、文件上传", 300, "#fed7aa", ["登录校验", "表单请求", "文件上传", "页面渲染", "权限控制"]),
        ("后端业务服务层", "auth / patient / case / import / prediction / report / log services", 480, "#e5e7eb", ["auth_service", "patient_service", "case_service", "import_service", "prediction_service", "report_service", "log_service"]),
        ("模型推理层", "real_model.py 适配 web_model 中的真实静态与动态模型", 660, "#e0e7ff", ["静态PGA-AMFormer", "动态D-PGA-AMFormer", "Scaler/Encoder", "缺失值容错预处理"]),
        ("数据持久层", "SQLite 自动初始化与本地数据持久化", 840, "#dcfce7", ["users", "patients", "cases", "predictions", "operation_logs", "model_versions"]),
        ("部署运行层", "阿里云 ECS：Nginx + Gunicorn + Flask + SQLite + web_model", 1020, "#f3f4f6", ["Nginx", "Gunicorn", "Flask应用", "database.db", "web_model模型文件"]),
    ]
    for name, desc, y, color, boxes in layers:
        s.text(92, y+58, name, 26, 700, anchor="start")
        s.rect(280, y, 1510, 115, color, "none", 16, 0)
        s.text(1035, y+33, desc, 20, 500, color="#374151")
        count = len(boxes)
        gap = 18
        bw = (1410 - gap*(count-1)) / count
        for i, b in enumerate(boxes):
            s.box(330+i*(bw+gap), y+54, bw, 45, b, COLORS["white"], "#94a3b8", 18, 500, 4)
        if y < 1020:
            s.arrow(1035, y+115, 1035, y+180)
    s.rect(1770, 510, 88, 420, "#99f6e4", "#5eead4", 24, 0)
    s.text(1814, 655, "操作\n日志\n审计", 28, 700, color="#0f766e")
    s.save(OUT_DIR / "fig6.3_system_architecture.svg")


def fig64():
    puml = puml_header("图6.4 系统业务流程图") + """
start
:医生登录系统;
:首页仪表盘;
if (是否已有患者?) then (否)
  :新增患者或Excel批量导入;
else (是)
  :查询并选择患者;
endif
:录入单次病例或动态批量导入;
:保存检测记录到cases表;
:调用静态PGA-AMFormer模型;
:保存静态预测到predictions表;
if (患者病例数 >= 2?) then (是)
  :读取同一患者历史病例;
  :按检查时间构造多时间点序列;
  :调用动态D-PGA-AMFormer模型;
  :保存动态预测结果;
else (否)
  :仅保留静态风险结果;
endif
:更新患者当前风险概率和风险等级;
if (病例数 >= 2?) then (是)
  :展示风险趋势图和指标趋势图;
else (否)
  :不展示趋势图，提示样本不足;
endif
:生成辅助决策报告并支持打印;
stop
@enduml
"""
    write(OUT_DIR / "fig6.4_system_business_process.puml", puml)
    s = Svg(1300, 1700, "临床辅助决策原型系统业务流程图")
    x = 650
    y = 115
    steps = [
        ("医生登录系统", 115),
        ("首页仪表盘", 230),
        ("新增/查询患者\n或Excel批量导入", 345),
        ("录入单次病例\n或动态批量导入", 475),
        ("保存检测记录到 cases 表", 610),
        ("调用静态 PGA-AMFormer\n生成基础风险", 745),
        ("保存静态预测到 predictions 表", 895),
    ]
    for i, (txt, yy) in enumerate(steps):
        s.box(x-210, yy, 420, 70 if "\n" not in txt else 96, txt, COLORS["gray"], "#4b5563", 24, 600, 24)
        if i < len(steps)-1:
            h = 70 if "\n" not in txt else 96
            s.arrow(x, yy+h, x, steps[i+1][1])
    s.rect(470, 1030, 360, 120, "#fff7ed", "#4b5563", 0, 3)
    s.text(650, 1082, "患者病例数 >= 2 ?", 25, 700)
    s.arrow(650, 965, 650, 1030)
    s.polyline([(470, 1090), (250, 1090), (250, 1240)], arrow=True, label="否")
    s.box(70, 1240, 360, 95, "仅保留静态风险\n不展示趋势图", COLORS["blue"], "#4b5563", 23, 600, 24)
    s.polyline([(830, 1090), (1060, 1090), (1060, 1200)], arrow=True, label="是")
    dyn = [
        ("读取历史病例", 1200),
        ("按检查时间构造\n多时间点序列", 1325),
        ("调用动态 D-PGA-AMFormer\n输出融合风险", 1470),
    ]
    for i, (txt, yy) in enumerate(dyn):
        s.box(880, yy, 360, 90, txt, COLORS["purple"], "#4b5563", 22, 600, 24)
        if i < len(dyn)-1:
            s.arrow(1060, yy+90, 1060, dyn[i+1][1])
    s.polyline([(250, 1335), (250, 1600), (650, 1600)], arrow=True)
    s.polyline([(1060, 1560), (1060, 1600), (650, 1600)], arrow=True)
    s.box(440, 1600, 420, 90, "更新患者当前风险状态\n生成报告并支持打印", COLORS["green"], "#4b5563", 23, 700, 24)
    s.save(OUT_DIR / "fig6.4_system_business_process.svg")


def fig65():
    puml = puml_header("图6.5 患者管理与详情页面原型结构图") + """
rectangle "左侧导航栏" as nav
rectangle "顶部用户信息栏" as top
rectangle "患者列表页 /patients" {
  [搜索：患者编号/姓名/风险等级]
  [新增患者]
  [Excel批量导入]
  [患者表格：编号、姓名、性别、年龄、科室、病例数、最近检查、风险等级、操作]
}
rectangle "患者详情页 /patients/{id}" {
  [患者基本信息卡片]
  [病例时间轴]
  [病例记录列表]
  [历史预测记录]
  [风险趋势图：病例数>=2时显示]
  [核心指标趋势图：病例数>=2时显示]
  [报告入口]
}
nav --> top
top --> "患者列表页 /patients"
"患者列表页 /patients" --> "患者详情页 /patients/{id}"
@enduml
"""
    write(OUT_DIR / "fig6.5_patient_management_page.puml", puml)
    s = Svg(1800, 1180, "患者管理与详情页面原型结构图")
    s.rect(70, 110, 220, 980, "#1e40af", "#1e40af", 0, 0)
    s.text(180, 170, "左侧导航栏", 25, 700, color="#ffffff")
    for i, item in enumerate(["首页", "患者管理", "新增病例", "批量导入", "模型版本", "操作日志"]):
        s.box(100, 230+i*90, 160, 50, item, "#dbeafe", "#bfdbfe", 18, 600, 6)
    s.rect(290, 110, 1440, 82, "#ffffff", "#e5e7eb", 0, 2)
    s.text(340, 160, "顶部用户信息栏：当前医生 / 科室 / 退出登录", 22, 500, anchor="start")
    s.rect(330, 225, 1360, 185, COLORS["gray"], "#e5e7eb", 12, 2)
    s.text(370, 275, "患者列表页 /patients", 27, 700, anchor="start")
    for i, item in enumerate(["患者编号", "姓名", "风险等级", "新增患者", "Excel批量导入"]):
        s.box(370+i*245, 315, 205, 52, item, COLORS["white"], "#cbd5e1", 18, 500, 6)
    s.rect(330, 445, 1360, 230, COLORS["white"], "#cbd5e1", 10, 2)
    headers = ["患者编号", "姓名", "性别", "年龄", "科室", "病例数", "最近检查", "风险等级", "操作"]
    for i, h in enumerate(headers):
        s.text(390+i*145, 500, h, 19, 700)
    for row in range(3):
        y = 545 + row*45
        s.line(350, y-24, 1670, y-24, color="#e5e7eb", sw=2)
        s.text(390, y, f"P2026{row+1:03d}", 18)
        s.text(535, y, "患者姓名", 18)
        s.text(680, y, "男/女/未知", 18)
        s.text(825, y, "56", 18)
        s.text(970, y, "神经内科", 18)
        s.text(1115, y, "1-4", 18)
        s.text(1260, y, "2026-05-19", 18)
        s.text(1405, y, "低/中/高", 18)
        s.text(1550, y, "详情/病例/报告", 18)
    s.rect(330, 720, 1360, 345, COLORS["blue"], "#cbd5e1", 12, 2)
    s.text(370, 770, "患者详情页 /patients/<patient_id>", 27, 700, anchor="start")
    detail_boxes = [
        (370, 815, 300, 90, "患者基本信息卡片"),
        (700, 815, 300, 90, "病例时间轴"),
        (1030, 815, 300, 90, "病例记录列表"),
        (1360, 815, 270, 90, "历史预测记录"),
        (370, 940, 580, 95, "风险趋势图\n仅当病例数 >= 2 显示"),
        (990, 940, 640, 95, "核心指标趋势图\n仅当病例数 >= 2 显示"),
    ]
    for x, y, w, h, txt in detail_boxes:
        s.box(x, y, w, h, txt, COLORS["white"], "#94a3b8", 21, 600, 8)
    s.save(OUT_DIR / "fig6.5_patient_management_page.svg")


def fig66():
    puml = puml_header("图6.6 系统接口调用流程图") + """
actor 临床医生 as doctor
participant "Web前端\\nJinja页面" as web
participant "Flask路由" as flask
participant "业务服务层" as service
database "SQLite数据库" as db
participant "real_model.py" as adapter
participant "PGA-AMFormer\\n静态模型" as static_model
participant "D-PGA-AMFormer\\n动态模型" as dynamic_model

doctor -> web : 登录/录入患者/上传Excel
web -> flask : POST /login 或 /patients/new 或 /patients/import
flask -> service : 校验字段、解析Excel
service -> db : 写入users/patients/cases/logs
db --> service : 返回主键
service --> web : 渲染页面

doctor -> web : 录入病例
web -> flask : POST /cases/new?patient_id=...
flask -> service : 保存病例并触发预测
service -> db : INSERT cases
service -> adapter : predict_static(case_data)
adapter -> static_model : 特征预处理 + 推理
static_model --> adapter : 静态风险概率
adapter --> service : 风险概率/等级/提示
service -> db : INSERT predictions(static)
service -> db : 查询患者病例数
alt 病例数 >= 2
  service -> adapter : predict_dynamic(case_list, latest_static_score)
  adapter -> dynamic_model : 时序特征 + 静态风险融合
  dynamic_model --> adapter : 动态修正量/融合风险
  adapter --> service : 动态风险结果
  service -> db : INSERT predictions(dynamic)
end
service -> db : UPDATE patients.current_risk
service --> web : 患者详情/报告/趋势图数据
web --> doctor : 展示预测结果与辅助决策报告
@enduml
"""
    write(OUT_DIR / "fig6.6_api_call_process.puml", puml)
    s = Svg(1850, 1600, "系统接口调用流程图")
    lanes = [
        ("临床医生", 100),
        ("Web前端\nJinja页面", 360),
        ("Flask路由", 620),
        ("业务服务层", 880),
        ("SQLite数据库", 1140),
        ("real_model.py", 1400),
        ("静态/动态模型", 1660),
    ]
    for name, x in lanes:
        s.box(x-85, 105, 170, 64, name, COLORS["gray"], "#94a3b8", 18, 700, 6)
        s.line(x, 170, x, 1515, dashed=True, sw=2, color="#94a3b8")
    def msg(x1, y, x2, label, dashed=False):
        s.arrow(x1, y, x2, y, dashed=dashed)
        s.text((x1+x2)/2, y-14, label, 18, 500)
    y = 240
    phases = [("患者信息管理与批量导入", y-70), ("病例录入与静态预测", 620), ("动态趋势预测与结果展示", 1090)]
    for title, yy in phases:
        s.rect(55, yy, 1740, 40, "#f8fafc", "#111827", 0, 2)
        s.text(925, yy+27, title, 21, 700)
    msg(100, y, 360, "登录/录入患者/上传Excel")
    msg(360, y+70, 620, "POST /login /patients/new /patients/import")
    msg(620, y+140, 880, "字段校验、Excel解析、权限校验")
    msg(880, y+210, 1140, "写入 patients / cases / operation_logs")
    msg(1140, y+280, 880, "返回主键", dashed=True)
    msg(880, y+350, 360, "渲染患者列表/详情", dashed=True)
    y = 700
    msg(100, y, 360, "录入单次检测病例")
    msg(360, y+70, 620, "POST /cases/new?patient_id=...")
    msg(620, y+140, 880, "保存病例并自动触发预测")
    msg(880, y+210, 1140, "INSERT cases")
    msg(880, y+280, 1400, "predict_static(case_data)")
    msg(1400, y+350, 1660, "特征预处理 + 静态推理")
    msg(1660, y+420, 1400, "静态风险概率", dashed=True)
    msg(1400, y+490, 880, "风险等级/关键因素/临床提示", dashed=True)
    msg(880, y+560, 1140, "INSERT predictions(static)")
    y = 1170
    msg(880, y, 1140, "查询患者病例数")
    s.box(720, y+55, 320, 72, "病例数 >= 2 ?", "#fff7ed", "#94a3b8", 21, 700, 8)
    msg(880, y+170, 1400, "predict_dynamic(case_list, static_score)")
    msg(1400, y+240, 1660, "时序特征 + 静态风险融合")
    msg(1660, y+310, 1400, "动态修正量/融合风险", dashed=True)
    msg(1400, y+380, 880, "动态预测结果", dashed=True)
    msg(880, y+450, 1140, "INSERT predictions(dynamic) / UPDATE patients")
    msg(880, y+520, 360, "患者详情、趋势图、报告数据", dashed=True)
    msg(360, y+590, 100, "展示预测结果与辅助决策报告", dashed=True)
    s.save(OUT_DIR / "fig6.6_api_call_process.svg")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig61()
    fig62()
    fig63()
    fig64()
    fig65()
    fig66()
    try:
        render_all_pngs()
    except ModuleNotFoundError:
        print("Pillow not available; SVG and PUML files were generated only.")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
