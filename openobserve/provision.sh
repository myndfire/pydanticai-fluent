#!/usr/bin/env bash
# Provision OpenObserve (streams are auto-created on first OTLP ingest) and
# import any dashboard bundles in ./dashboards/*.json.
#
# Mirrors kibana/provision-dashboards.sh: waits for the UI, verifies the
# telemetry streams exist, imports dashboards, then prints useful URLs.
#
# Usage:
#   ./openobserve/provision.sh
#   OPENOBSERVE_URL=http://localhost:5080 ./openobserve/provision.sh

set -euo pipefail

OPENOBSERVE_URL="${OPENOBSERVE_URL:-${OPENOBSERVE_UI_URL:-http://localhost:5080}}"
ORG="${OPENOBSERVE_ORG:-default}"
EMAIL="${ZO_ROOT_USER_EMAIL:-admin@example.com}"
PASSWORD="${ZO_ROOT_USER_PASSWORD:-Admin1234!}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARDS_DIR="${SCRIPT_DIR}/dashboards"
ENV_FILE="${SCRIPT_DIR}/../.env"

# Read a single key from .env without sourcing it (values may contain spaces).
read_env_var() {
  local key="$1" file="$2" default="$3"
  local val=""
  if [[ -f "${file}" ]]; then
    val="$(grep -E "^${key}=" "${file}" | head -n1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
  fi
  if [[ -z "${val}" ]]; then
    val="${default}"
  fi
  printf '%s' "${val}"
}

OPENOBSERVE_URL="${OPENOBSERVE_URL:-$(read_env_var OPENOBSERVE_UI_URL "${ENV_FILE}" "http://localhost:5080")}"
ORG="${OPENOBSERVE_ORG:-$(read_env_var OPENOBSERVE_ORG "${ENV_FILE}" "default")}"
EMAIL="$(read_env_var ZO_ROOT_USER_EMAIL "${ENV_FILE}" "${EMAIL:-admin@example.com}")"
PASSWORD="$(read_env_var ZO_ROOT_USER_PASSWORD "${ENV_FILE}" "${PASSWORD:-Admin1234!}")"

AUTH=(-u "${EMAIL}:${PASSWORD}")

echo "==> Waiting for OpenObserve at ${OPENOBSERVE_URL} ..."
for i in $(seq 1 60); do
  if curl -fsS -o /dev/null "${OPENOBSERVE_URL}/web/" 2>/dev/null; then
    echo "    OpenObserve is available (attempt ${i})"
    break
  fi
  if [[ "${i}" == "60" ]]; then
    echo "ERROR: OpenObserve did not become available in time." >&2
    exit 1
  fi
  sleep 2
done

echo "==> Verifying telemetry streams (auto-created on first OTLP ingest) ..."
streams_json="$(curl -fsS "${AUTH[@]}" "${OPENOBSERVE_URL}/api/${ORG}/streams" || true)"
python3 - "${streams_json}" <<'PY'
import json
import sys

raw = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    data = json.loads(raw)
except Exception:
    print("    (could not read stream list; it populates after the first export)")
    raise SystemExit(0)

found = {"logs": False, "traces": False, "metrics": False}
for stream in data.get("list", []):
    stype = stream.get("stream_type")
    name = stream.get("name", "")
    if stype in found and name == "agent_harness":
        found[stype] = True
    if stype == "metrics":
        found["metrics"] = True

for signal, present in found.items():
    print(f"    {signal:8s}: {'present' if present else 'not yet (waiting for data)'}")
PY

if [[ -d "${DASHBOARDS_DIR}" ]] && compgen -G "${DASHBOARDS_DIR}/*.json" > /dev/null; then
  echo "==> Importing OpenObserve dashboards (replace by title) ..."
  existing_json="$(curl -fsS "${AUTH[@]}" "${OPENOBSERVE_URL}/api/${ORG}/dashboards" || echo '{"dashboards":[]}')"
  shopt -s nullglob
  for dashboard in "${DASHBOARDS_DIR}"/*.json; do
    name="$(basename "${dashboard}")"
    title="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["title"])' "${dashboard}")"
    existing_id="$(
      python3 -c '
import json, sys
title = sys.argv[1]
data = json.loads(sys.argv[2] or "{}")
for d in data.get("dashboards", []):
    if d.get("title") != title:
        continue
    nested = d.get("v8") if isinstance(d.get("v8"), dict) else {}
    print(
        d.get("dashboard_id")
        or d.get("dashboardId")
        or d.get("id")
        or nested.get("dashboardId")
        or ""
    )
    break
' "${title}" "${existing_json}"
    )"
    if [[ -n "${existing_id}" ]]; then
      echo "    -> ${name}: replacing existing '${title}' (${existing_id})"
      curl -fsS "${AUTH[@]}" -X DELETE \
        "${OPENOBSERVE_URL}/api/${ORG}/dashboards/${existing_id}" > /dev/null || true
    else
      echo "    -> ${name}: creating '${title}'"
    fi
    status="$(curl -s -o /tmp/o2_dash_import.out -w '%{http_code}' \
      "${AUTH[@]}" -X POST \
      "${OPENOBSERVE_URL}/api/${ORG}/dashboards" \
      -H 'Content-Type: application/json' \
      --data-binary "@${dashboard}")"
    if [[ "${status}" != "200" && "${status}" != "201" ]]; then
      echo "ERROR: dashboard import failed (${status}): $(head -c 500 /tmp/o2_dash_import.out)" >&2
      exit 1
    fi
  done
  shopt -u nullglob
else
  echo "==> No OpenObserve dashboard bundles found (${DASHBOARDS_DIR}); skipping import."
fi

echo
echo "==> Done. OpenObserve:"
echo "    UI:      ${OPENOBSERVE_URL}/web/"
echo "    Logs:    ${OPENOBSERVE_URL}/web/logs?org_identifier=${ORG}"
echo "    Traces:  ${OPENOBSERVE_URL}/web/traces?org_identifier=${ORG}"
echo "    Metrics: ${OPENOBSERVE_URL}/web/metrics?org_identifier=${ORG}"
echo "    Login:   ${EMAIL} / ${PASSWORD}"
