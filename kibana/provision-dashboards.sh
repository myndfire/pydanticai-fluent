#!/usr/bin/env bash
# Provision all Kibana dashboards and print dashboard links automatically.
# Compatible with macOS Bash 3.2 (no mapfile).
#
# Usage:
#   ./kibana/provision-dashboards.sh
#   KIBANA_URL=http://localhost:5601 ./kibana/provision-dashboards.sh

set -euo pipefail

KIBANA_URL="${KIBANA_URL:-http://localhost:5601}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVED_OBJECTS_DIR="${SCRIPT_DIR}/saved-objects"
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

# Browser-accessible Langfuse UI base + project id, used to turn the log
# document's trace_id into a clickable link to the trace where it happened.
# Environment values win; otherwise fall back to .env, then sane defaults.
LANGFUSE_UI_URL="${LANGFUSE_UI_URL:-$(read_env_var LANGFUSE_UI_URL "${ENV_FILE}" "http://localhost:3000")}"
LANGFUSE_PROJECT_ID="${LANGFUSE_PROJECT_ID:-$(read_env_var LANGFUSE_PROJECT_ID "${ENV_FILE}" "local-project")}"
TRACE_LINK_BASE="${LANGFUSE_UI_URL%/}/project/${LANGFUSE_PROJECT_ID}/traces/"

LOGS_DATA_VIEW_ID="1ee66b57-99f5-44bd-9828-5b690f3cc8af"
LOGS_DATA_VIEW_TITLE="logs-generic.otel-default*"

TRACES_DATA_VIEW_ID="7f6f1a30-63b2-4c9b-8b0c-3f8bfe8d9a10"
TRACES_DATA_VIEW_TITLE="traces-generic.otel-default*"

METRICS_DATA_VIEW_ID="a3c1d2e4-5b6f-47a8-9c0d-1e2f3a4b5c6d"
METRICS_DATA_VIEW_TITLE="metrics-generic.otel-default*"

echo "==> Waiting for Kibana at ${KIBANA_URL} ..."
for i in $(seq 1 120); do
  status="$(
    curl -fsS "${KIBANA_URL}/api/status" 2>/dev/null |
      python3 -c 'import sys,json; print(json.load(sys.stdin)["status"]["overall"]["level"])' 2>/dev/null ||
      true
  )"

  if [[ "${status}" == "available" ]]; then
    echo "    Kibana is available (attempt ${i})"
    break
  fi

  if [[ "${i}" == "120" ]]; then
    echo "ERROR: Kibana did not become available in time." >&2
    exit 1
  fi

  sleep 2
done

ensure_data_view() {
  dv_id="$1"
  title="$2"
  trace_link_base="${3:-}"

  echo "==> Ensuring data view '${title}' exists ..."

  python3 - "$KIBANA_URL" "$dv_id" "$title" "$trace_link_base" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base, dv_id, title = sys.argv[1:4]
trace_link_base = sys.argv[4] if len(sys.argv) > 4 else ""

# Optional: render trace_id as a "View in Langfuse" link.
extra = {}
if trace_link_base:
    extra["fieldFormats"] = {
        "trace_id": {
            "id": "url",
            "params": {
                "type": "a",
                "urlTemplate": trace_link_base + "{{value}}",
                "labelTemplate": "View in Langfuse",
            },
        }
    }

headers = {"Content-Type": "application/json", "kbn-xsrf": "true"}

def request(method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

status, payload = request("GET", f"{base}/api/data_views/data_view/{dv_id}")

if status == 200:
    body = {
        "data_view": {
            "title": title,
            "timeFieldName": "@timestamp",
            "name": title,
            **extra,
        },
        "refresh_fields": True,
    }
    status, payload = request("POST", f"{base}/api/data_views/data_view/{dv_id}", body)
    if status != 200:
        print(f"ERROR: failed to update data view ({status}): {payload[:1000]}", file=sys.stderr)
        sys.exit(1)
    print(f"    updated data view: {dv_id}")

elif status == 404:
    body = {
        "data_view": {
            "id": dv_id,
            "title": title,
            "timeFieldName": "@timestamp",
            "name": title,
            **extra,
        },
        "override": True,
    }
    status, payload = request("POST", f"{base}/api/data_views/data_view", body)
    if status != 200:
        print(f"ERROR: failed to create data view ({status}): {payload[:1000]}", file=sys.stderr)
        sys.exit(1)
    print(f"    created data view: {dv_id}")

else:
    print(f"ERROR: failed to check data view ({status}): {payload[:1000]}", file=sys.stderr)
    sys.exit(1)

status, payload = request("GET", f"{base}/api/data_views/data_view/{dv_id}")
if status != 200:
    print(f"ERROR: data view verification failed ({status}): {payload[:1000]}", file=sys.stderr)
    sys.exit(1)

dv = json.loads(payload).get("data_view", {})
print(f"    verified data view: {dv.get('id')}")
print(f"    title: {dv.get('title')}")
print(f"    time field: {dv.get('timeFieldName')}")
PY
}

ensure_data_view "${LOGS_DATA_VIEW_ID}" "${LOGS_DATA_VIEW_TITLE}" "${TRACE_LINK_BASE}"
ensure_data_view "${TRACES_DATA_VIEW_ID}" "${TRACES_DATA_VIEW_TITLE}"
ensure_data_view "${METRICS_DATA_VIEW_ID}" "${METRICS_DATA_VIEW_TITLE}"

if [[ ! -d "${SAVED_OBJECTS_DIR}" ]]; then
  echo "ERROR: saved-objects directory not found: ${SAVED_OBJECTS_DIR}" >&2
  exit 1
fi

NDJSON_LIST_FILE="$(mktemp)"
trap 'rm -f "${NDJSON_LIST_FILE}"' EXIT

find "${SAVED_OBJECTS_DIR}" -maxdepth 1 -type f -name '*.ndjson' | sort > "${NDJSON_LIST_FILE}"

NDJSON_COUNT="$(wc -l < "${NDJSON_LIST_FILE}" | tr -d ' ')"

if [[ "${NDJSON_COUNT}" -eq 0 ]]; then
  echo "ERROR: no .ndjson files found in ${SAVED_OBJECTS_DIR}" >&2
  exit 1
fi

echo "==> Importing ${NDJSON_COUNT} saved-object bundle(s) ..."

while IFS= read -r ndjson; do
  [[ -z "${ndjson}" ]] && continue

  echo "    -> $(basename "${ndjson}")"

  result="$(
    curl -fsS -X POST \
      "${KIBANA_URL}/api/saved_objects/_import?overwrite=true" \
      -H "kbn-xsrf: true" \
      --form "file=@${ndjson}"
  )"

  python3 - "$result" "$(basename "${ndjson}")" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
filename = sys.argv[2]

if not payload.get("success"):
    print(f"ERROR: import failed for {filename}:", json.dumps(payload, indent=2), file=sys.stderr)
    sys.exit(1)

results = payload.get("successResults", [])
print(f"       imported {len(results)} saved objects")
for item in results:
    print(f"         - {item['type']}: {item['id']}")
PY

done < "${NDJSON_LIST_FILE}"

echo
echo "==> All Kibana saved objects provisioned successfully."
echo "    Saved-object directory: ${SAVED_OBJECTS_DIR}"

echo
echo "==> Done. Open the dashboards:"
echo

# Discover dashboard objects directly from the same NDJSON files we imported.
# This prevents the printed list from becoming stale as dashboards are added.
python3 - "${KIBANA_URL}" "${SAVED_OBJECTS_DIR}" <<'PY'
import glob
import json
import os
import sys

base = sys.argv[1].rstrip("/")
directory = sys.argv[2]

dashboards = {}

for path in sorted(glob.glob(os.path.join(directory, "*.ndjson"))):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") != "dashboard":
                continue

            dashboard_id = obj.get("id")
            title = obj.get("attributes", {}).get("title", dashboard_id)

            if dashboard_id:
                dashboards[dashboard_id] = title

if not dashboards:
    print("    No dashboard objects found.")
    sys.exit(0)

for dashboard_id, title in sorted(dashboards.items(), key=lambda x: x[1].lower()):
    print(f"    {title}:")
    print(f"    {base}/app/dashboards#/view/{dashboard_id}")
    print()
PY
