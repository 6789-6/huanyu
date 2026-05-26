#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

install -d /var/www/huanyu
install -d /opt/huanyu/backend
install -d /opt/huanyu/demo_assets
install -d /opt/huanyu/shouyu_project

rsync -a --delete "$ROOT/runtime/frontend/frontweb/" /var/www/huanyu/
rsync -a --delete "$ROOT/runtime/backend/algorithm_service/" /opt/huanyu/backend/algorithm_service/
rsync -a --delete "$ROOT/runtime/demo_assets/" /opt/huanyu/demo_assets/
rsync -a --delete "$ROOT/ml/shouyu_project/" /opt/huanyu/shouyu_project/

python3 -m venv /opt/huanyu/backend/venv
/opt/huanyu/backend/venv/bin/python -m pip install --upgrade pip==21.3.1 setuptools==59.6.0 wheel==0.37.1
/opt/huanyu/backend/venv/bin/pip install -r "$ROOT/deploy/backend/requirements-py36.txt"

cp "$ROOT/deploy/backend/huanyu-backend.service" /etc/systemd/system/huanyu-backend.service
cp "$ROOT/deploy/nginx/huanyu.conf" /etc/nginx/conf.d/huanyu.conf

systemctl daemon-reload
systemctl enable huanyu-backend
systemctl restart huanyu-backend
nginx -t
systemctl restart nginx

echo "Huanyu lite deployment complete."
echo "Health: http://SERVER_IP/api/health"
