#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
export OPEN_ROBO_NOVNC=1

Xvfb "${DISPLAY}" -screen 0 1920x1080x24 &
sleep 1

fluxbox &
sleep 1
xsetroot -solid '#3d3d3d' 2>/dev/null || true

cd /app
python3 -u -m on_device_app &

x11vnc -display "${DISPLAY}" -forever -shared -rfbport 5900 -nopw -listen 0.0.0.0 -noxdamage &

NOVNC_WEB="${NOVNC_WEB:-/usr/share/novnc}"
if [[ ! -d "${NOVNC_WEB}" ]]; then
  NOVNC_WEB="/usr/share/webapps/novnc"
fi
exec python3 -m websockify --web="${NOVNC_WEB}" 0.0.0.0:6080 localhost:5900
