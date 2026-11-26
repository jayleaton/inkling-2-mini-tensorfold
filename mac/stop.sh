#!/usr/bin/env bash
# Stop the Metal tensorfold-native server started by mac/start.sh.
# Usage: mac/stop.sh      Env: PORT, STOP_TIMEOUT (default 30)
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=scripts/config.sh
source ./scripts/config.sh

if [[ ! -f "$PID_FILE" ]]; then
  log "No PID file ($PID_FILE): nothing to stop"
  exit 0
fi

pid=$(cat "$PID_FILE" 2>/dev/null || true)
if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  log "Server (pid ${pid:-?}) is not running; cleared stale PID file"
  exit 0
fi

api_host="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && api_host=127.0.0.1
[[ "$api_host" == *:* ]] && api_host="[$api_host]"
busy=$(curl -s --max-time 3 "http://$api_host:$PORT/health" 2>/dev/null |
       python3 -c 'import json,sys; h=json.load(sys.stdin); n=h.get("running", h.get("requests_running", 0)); print(n if type(n) is int and n >= 0 else 0)' 2>/dev/null || echo 0)
[[ "$busy" =~ ^[0-9]+$ ]] || busy=0
(( busy == 0 )) || warn "$busy request(s) still running will be cut off"

timeout="${STOP_TIMEOUT:-30}"
log "Stopping tensorfold-native (pid $pid, up to ${timeout}s)"
kill -TERM "$pid" 2>/dev/null || true
for (( i = 0; i < timeout; i++ )); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$pid" 2>/dev/null; then
  warn "Still running after ${timeout}s; sending SIGKILL"
  kill -KILL "$pid" 2>/dev/null || true
fi
rm -f "$PID_FILE"
log "Stopped tensorfold-native"
