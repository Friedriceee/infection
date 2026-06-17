#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="/opt/cns-cdss"
APP_USER="cns"
SERVICE_NAME="cns-cdss"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "请使用 root 用户执行此脚本"
  exit 1
fi

if ! id "${APP_USER}" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "${APP_USER}"
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y python3.9 python3.9-venv python3.9-dev python3-pip nginx sqlite3
elif command -v yum >/dev/null 2>&1; then
  yum install -y python3 python3-pip nginx sqlite
else
  echo "未识别的 Linux 包管理器，请手动安装 python3、python3-venv、pip、nginx、sqlite3"
  exit 1
fi

cd "${APP_ROOT}"
if command -v python3.9 >/dev/null 2>&1; then
  PYTHON_BIN="python3.9"
else
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" -m venv venv
"${APP_ROOT}/venv/bin/pip" install --upgrade pip
grep -v '^torch' "${APP_ROOT}/cns_prediction_system/requirements.txt" > /tmp/cns-req-no-torch.txt
"${APP_ROOT}/venv/bin/pip" install -r /tmp/cns-req-no-torch.txt
"${APP_ROOT}/venv/bin/pip" install filelock typing-extensions sympy networkx fsspec -i http://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com
"${APP_ROOT}/venv/bin/pip" install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --no-deps

chown -R "${APP_USER}:${APP_USER}" "${APP_ROOT}"

install -m 0644 "${APP_ROOT}/cns_prediction_system/deploy/cns-cdss.service" "/etc/systemd/system/${SERVICE_NAME}.service"
install -m 0644 "${APP_ROOT}/cns_prediction_system/deploy/nginx_cns_cdss.conf" "/etc/nginx/conf.d/cns-cdss.conf"

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"
nginx -t
systemctl enable nginx
systemctl restart nginx

echo "部署完成：请访问 http://116.62.66.20"
