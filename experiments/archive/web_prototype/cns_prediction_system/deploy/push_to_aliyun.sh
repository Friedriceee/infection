#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-116.62.66.20}"
USER_NAME="${2:-root}"
PACKAGE="${3:-/private/tmp/cns-cdss-deploy.tar.gz}"

if [[ ! -f "${PACKAGE}" ]]; then
  echo "未找到部署包：${PACKAGE}"
  exit 1
fi

scp "${PACKAGE}" "${USER_NAME}@${HOST}:/tmp/cns-cdss-deploy.tar.gz"
ssh "${USER_NAME}@${HOST}" '
  set -euo pipefail
  mkdir -p /opt/cns-cdss
  tar -xzf /tmp/cns-cdss-deploy.tar.gz -C /opt/cns-cdss
  bash /opt/cns-cdss/cns_prediction_system/deploy/install_on_aliyun.sh
'

echo "部署完成：http://${HOST}"

