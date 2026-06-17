from datetime import datetime
import json
import os
from functools import wraps

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

import config
from models.db import close_db, init_db
from models import case_model, model_version_model, patient_model, prediction_model, user_model
from services import auth_service, case_service, import_service, log_service, patient_service, prediction_service, report_service
from utils.import_parser import parse_uploaded_records
from utils.risk_level import risk_badge_class
from utils.validators import validate_patient_payload


def create_app():
    app = Flask(__name__)
    app.config.from_object(config)
    app.teardown_appcontext(close_db)

    with app.app_context():
        os.makedirs(os.path.dirname(app.config["DATABASE"]), exist_ok=True)
        init_db()

    @app.context_processor
    def inject_globals():
        user = user_model.get_user_by_id(session["user_id"]) if session.get("user_id") else None
        return {
            "current_user": user,
            "risk_badge_class": risk_badge_class,
            "system_name": "中枢神经系统感染风险预测临床辅助决策系统",
        }

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not session.get("user_id"):
                return redirect(url_for("login", next=request.path))
            return view(*args, **kwargs)

        return wrapped

    def admin_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = user_model.get_user_by_id(session.get("user_id"))
            if not user or user["role"] != "admin":
                flash("仅管理员可以访问该页面", "error")
                return redirect(url_for("dashboard"))
            return view(*args, **kwargs)

        return wrapped

    def doctor_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = user_model.get_user_by_id(session.get("user_id"))
            if not user or user["role"] != "doctor":
                flash("管理员账号仅用于系统管理，不能访问临床预测页面", "error")
                return redirect(url_for("admin"))
            return view(*args, **kwargs)

        return wrapped

    def current_user_id():
        return session.get("user_id")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            user = auth_service.authenticate(username, password)
            if user:
                session.clear()
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                session["role"] = user["role"]
                log_service.write_log("登录", "user", user["id"], "用户登录系统")
                if user["role"] == "admin":
                    return redirect(url_for("admin"))
                return redirect(request.args.get("next") or url_for("dashboard"))
            flash("用户名或密码错误", "error")
        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        log_service.write_log("退出", "user", current_user_id(), "用户退出系统")
        session.clear()
        return redirect(url_for("login"))

    @app.route("/")
    @login_required
    @doctor_required
    def dashboard():
        today = datetime.now().strftime("%Y-%m-%d")
        stats = patient_service.build_dashboard_stats(today)
        high_risk_patients = patient_model.list_high_risk_patients(8)
        return render_template(
            "dashboard.html",
            stats=stats,
            high_risk_patients=high_risk_patients,
        )

    @app.route("/patients")
    @login_required
    @doctor_required
    def patients():
        keyword = request.args.get("keyword", "").strip()
        risk_level = request.args.get("risk_level", "").strip()
        rows = patient_model.list_patients(keyword, risk_level)
        return render_template("patients.html", patients=rows, keyword=keyword, selected_risk=risk_level)

    @app.route("/patients/new", methods=["GET", "POST"])
    @login_required
    @doctor_required
    def patient_new():
        if request.method == "POST":
            data = {key: request.form.get(key, "").strip() for key in request.form.keys()}
            data["department"] = "神经内科"
            existing = patient_model.get_patient_by_no_any_status(data.get("patient_no"))
            active_existing = existing if existing and not existing["is_deleted"] else None
            errors = validate_patient_payload(data, active_existing)
            if not errors:
                if existing and existing["is_deleted"]:
                    patient_id = existing["id"]
                    patient_model.restore_patient(patient_id, data)
                    log_service.write_log("恢复患者", "patient", patient_id, f"恢复患者 {data.get('name')}")
                    flash("该患者编号曾被删除，已恢复患者信息", "success")
                else:
                    patient_id = patient_model.create_patient(data)
                    log_service.write_log("新增患者", "patient", patient_id, f"新增患者 {data.get('name')}")
                    flash("患者信息保存成功", "success")
                return redirect(url_for("patient_detail", patient_id=patient_id))
            for error in errors:
                flash(error, "error")
            return render_template("patient_form.html", patient=data)
        return render_template("patient_form.html", patient={})

    @app.route("/patients/import", methods=["GET", "POST"])
    @login_required
    @doctor_required
    def patient_import():
        summary = None
        if request.method == "POST":
            upload = request.files.get("file")
            if not upload or not upload.filename:
                flash("请选择需要导入的 Excel 或 CSV 文件", "error")
                return render_template("patient_import.html", summary=summary)
            try:
                records = parse_uploaded_records(upload)
                summary = import_service.import_patient_records(records, current_user_id())
                log_service.write_log(
                    "批量导入",
                    "patient",
                    None,
                    f"导入患者 {summary['patients_created']} 个，病例 {summary['cases_created']} 条",
                )
                flash("批量导入完成", "success")
            except Exception as error:
                flash(str(error), "error")
        return render_template("patient_import.html", summary=summary)

    @app.route("/patients/<int:patient_id>")
    @login_required
    @doctor_required
    def patient_detail(patient_id):
        patient = patient_model.get_patient(patient_id)
        if not patient:
            flash("患者不存在", "error")
            return redirect(url_for("patients"))

        cases = case_model.list_cases_by_patient(patient_id)
        predictions = prediction_model.list_predictions_by_patient(patient_id)
        chart_data = report_service.build_patient_chart_data(cases, predictions)
        static_cases = {
            item["case_id"]: True
            for item in predictions
            if item["prediction_type"] == "static" and item["case_id"]
        }
        dynamic_cases = {
            item["case_id"]: True
            for item in predictions
            if item["prediction_type"] == "dynamic" and item["case_id"]
        }
        return render_template(
            "patient_detail.html",
            patient=patient,
            cases=cases,
            predictions=predictions,
            chart_data=chart_data,
            static_cases=static_cases,
            dynamic_cases=dynamic_cases,
        )

    @app.route("/patients/<int:patient_id>/delete", methods=["POST"])
    @login_required
    @doctor_required
    def patient_delete(patient_id):
        patient = patient_model.get_patient(patient_id)
        if not patient:
            flash("患者不存在", "error")
            return redirect(url_for("patients"))
        patient_model.soft_delete_patient(patient_id)
        log_service.write_log("删除患者", "patient", patient_id, f"删除/出院患者 {patient['name']}")
        flash("患者已删除，相关病例和预测记录已作废", "success")
        return redirect(url_for("patients"))

    @app.route("/cases/new", methods=["GET", "POST"])
    @login_required
    @doctor_required
    def case_new():
        patient_id = request.args.get("patient_id", type=int) or request.form.get("patient_id", type=int)
        patient = patient_model.get_patient(patient_id) if patient_id else None
        if not patient:
            flash("请先选择患者后再新增病例", "error")
            return redirect(url_for("patients"))

        if request.method == "POST":
            if not request.form.get("visit_time"):
                flash("检查时间不能为空", "error")
                return render_template("case_form.html", patient=patient, case={}, mode="new")
            case_id = case_service.create_case_from_form(request.form, patient_id, current_user_id())
            prediction_service.run_auto_prediction_after_case(case_id, current_user_id())
            log_service.write_log("新增病例并自动预测", "case", case_id, f"患者 {patient['name']} 新增病例后完成预测")
            flash("病例保存成功，系统已自动完成风险预测", "success")
            return redirect(url_for("patient_detail", patient_id=patient_id))

        return render_template("case_form.html", patient=patient, case={}, mode="new")

    @app.route("/cases/<int:case_id>")
    @login_required
    @doctor_required
    def case_detail(case_id):
        case = case_model.get_case(case_id)
        if not case:
            flash("病例不存在", "error")
            return redirect(url_for("patients"))
        patient = patient_model.get_patient(case["patient_id"])
        static_prediction = next(iter(prediction_model.list_predictions_for_case(case_id, "static")), None)
        dynamic_prediction = next(iter(prediction_model.list_predictions_for_case(case_id, "dynamic")), None)
        return render_template(
            "case_detail.html",
            case=case,
            patient=patient,
            static_prediction=static_prediction,
            dynamic_prediction=dynamic_prediction,
        )

    @app.route("/cases/<int:case_id>/edit", methods=["GET", "POST"])
    @login_required
    @doctor_required
    def case_edit(case_id):
        case = case_model.get_case(case_id)
        if not case:
            flash("病例不存在", "error")
            return redirect(url_for("patients"))
        patient = patient_model.get_patient(case["patient_id"])
        if request.method == "POST":
            if not request.form.get("visit_time"):
                flash("检查时间不能为空", "error")
                return render_template("case_form.html", patient=patient, case=case_service.row_to_dict(case), mode="edit")
            case_service.update_case_from_form(case_id, request.form, patient["id"], current_user_id())
            prediction_service.rerun_case_prediction(case_id, current_user_id())
            log_service.write_log("编辑病例并重新预测", "case", case_id, f"编辑患者 {patient['name']} 病例")
            flash("病例已更新，并已重新生成预测结果", "success")
            return redirect(url_for("case_detail", case_id=case_id))
        return render_template("case_form.html", patient=patient, case=case_service.row_to_dict(case), mode="edit")

    @app.route("/cases/<int:case_id>/repredict", methods=["POST"])
    @login_required
    @doctor_required
    def case_repredict(case_id):
        case = case_model.get_case(case_id)
        if not case:
            flash("病例不存在", "error")
            return redirect(url_for("patients"))
        prediction_service.rerun_case_prediction(case_id, current_user_id())
        log_service.write_log("重新预测", "case", case_id, "手动重新执行风险预测")
        flash("已重新生成预测结果", "success")
        return redirect(url_for("case_detail", case_id=case_id))

    @app.route("/cases/<int:case_id>/delete", methods=["POST"])
    @login_required
    @doctor_required
    def case_delete(case_id):
        case = case_model.get_case(case_id)
        if not case:
            flash("病例不存在", "error")
            return redirect(url_for("patients"))
        patient_id = case["patient_id"]
        case_model.soft_delete_case(case_id)
        prediction_model.invalidate_by_case(case_id)
        prediction_service.refresh_patient_current_risk(patient_id)
        log_service.write_log("删除病例", "case", case_id, f"删除患者 {case['patient_name']} 病例")
        flash("病例已删除，关联预测记录已作废", "success")
        return redirect(url_for("patient_detail", patient_id=patient_id))

    @app.route("/predictions/<int:prediction_id>")
    @login_required
    @doctor_required
    def prediction_detail(prediction_id):
        prediction = prediction_model.get_prediction(prediction_id)
        if not prediction:
            flash("预测记录不存在", "error")
            return redirect(url_for("patients"))
        result = prediction_service.parse_result(prediction)
        input_snapshot = json.loads(prediction["input_snapshot"]) if prediction["input_snapshot"] else {}
        return render_template(
            "prediction_detail.html",
            prediction=prediction,
            result=result,
            input_snapshot=input_snapshot,
        )

    @app.route("/report/<int:patient_id>")
    @login_required
    @doctor_required
    def report(patient_id):
        patient = patient_model.get_patient(patient_id)
        if not patient:
            flash("患者不存在", "error")
            return redirect(url_for("patients"))
        report_data = report_service.build_report(patient)
        model_versions = model_version_model.list_model_versions()
        log_service.write_log("查看报告", "patient", patient_id, f"查看患者 {patient['name']} 辅助决策报告")
        return render_template("report.html", patient=patient, report=report_data, model_versions=model_versions)

    @app.route("/model_versions")
    @login_required
    def model_versions():
        rows = model_version_model.list_model_versions()
        return render_template("model_versions.html", model_versions=rows)

    @app.route("/model_versions/new", methods=["GET", "POST"])
    @login_required
    @admin_required
    def model_version_new():
        if request.method == "POST":
            data = {key: request.form.get(key, "").strip() for key in request.form.keys()}
            if not data.get("model_name") or not data.get("model_version") or data.get("model_type") not in ("static", "dynamic"):
                flash("模型名称、版本和类型不能为空", "error")
                return render_template("model_version_form.html", model=data)
            version_id = model_version_model.create_model_version(data)
            if data.get("status") == "enabled":
                model_version_model.set_enabled(version_id)
            log_service.write_log("新增模型版本", "model_version", version_id, f"新增模型版本 {data.get('model_name')} {data.get('model_version')}")
            flash("模型版本已新增", "success")
            return redirect(url_for("model_versions"))
        return render_template("model_version_form.html", model={})

    @app.route("/model_versions/<int:version_id>/enable", methods=["POST"])
    @login_required
    @admin_required
    def model_version_enable(version_id):
        if model_version_model.set_enabled(version_id):
            log_service.write_log("启用模型版本", "model_version", version_id, "切换当前启用模型版本")
            flash("模型版本已设为当前启用", "success")
        else:
            flash("模型版本不存在", "error")
        return redirect(url_for("model_versions"))

    @app.route("/model_versions/<int:version_id>/delete", methods=["POST"])
    @login_required
    @admin_required
    def model_version_delete(version_id):
        row = model_version_model.get_model_version(version_id)
        if not row:
            flash("模型版本不存在", "error")
            return redirect(url_for("model_versions"))
        if row["status"] == "enabled":
            flash("当前启用版本不能删除，请先启用同类型其他版本", "error")
            return redirect(url_for("model_versions"))
        model_version_model.delete_model_version(version_id)
        log_service.write_log("删除模型版本", "model_version", version_id, f"删除模型版本 {row['model_name']} {row['model_version']}")
        flash("模型版本已删除", "success")
        return redirect(url_for("model_versions"))

    @app.route("/logs")
    @login_required
    @admin_required
    def logs():
        rows = log_model_rows()
        return render_template("logs.html", logs=rows)

    def log_model_rows():
        from models import log_model

        return log_model.list_logs()

    @app.route("/admin")
    @login_required
    @admin_required
    def admin():
        return render_template("admin.html")

    @app.route("/admin/users")
    @login_required
    @admin_required
    def users():
        rows = user_model.list_users()
        return render_template("users.html", users=rows)

    @app.route("/admin/users/new", methods=["GET", "POST"])
    @login_required
    @admin_required
    def user_new():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            role = request.form.get("role", "doctor")
            if not username or not password:
                flash("用户名和密码不能为空", "error")
                return render_template("user_form.html", user=request.form.to_dict())
            if role not in ("doctor", "admin"):
                flash("角色类型不正确", "error")
                return render_template("user_form.html", user=request.form.to_dict())
            if user_model.get_user_by_username(username):
                flash("用户名已存在", "error")
                return render_template("user_form.html", user=request.form.to_dict())
            user_id = user_model.create_user(
                {
                    "username": username,
                    "password": password,
                    "role": role,
                    "real_name": request.form.get("real_name", "").strip(),
                    "department": request.form.get("department", "").strip(),
                }
            )
            log_service.write_log("新增用户", "user", user_id, f"管理员创建账号 {username}")
            flash("账号创建成功", "success")
            return redirect(url_for("users"))
        return render_template("user_form.html", user={})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
