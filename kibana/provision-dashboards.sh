#!/usr/bin/env bash
# Provision all Kibana saved-object bundles in kibana/saved-objects/.
#
# Behavior:
#   1. Wait for Kibana.
#   2. Ensure the shared logs data view exists with the exact stable ID.
#   3. Import every *.ndjson file in saved-objects/ with overwrite=true.
#
# This keeps dashboard provisioning generic: adding a new dashboard only
# requires dropping another .ndjson file into kibana/saved-objects/.
#
# Requires: curl, python3, running Kibana.
#
# Usage:
#   ./kibana/provision-dashboards.sh
#   KIBANA_URL=http://localhost:5601 ./kibana/provision-dashboards.sh

set -euo pipefail

KIBANA_URL="${KIBANA_URL:-http://localhost:5601}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVED_OBJECTS_DIR="${SCRIPT_DIR}/saved-objects"

DATA_VIEW_TITLE="logs-generic.otel-default*"
DATA_VIEW_ID="1ee66b57-99f5-44bd-9828-5b690f3cc8af"

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

echo "==> Ensuring data view '${DATA_VIEW_TITLE}' exists ..."

python3 - "$KIBANA_URL" "$DATA_VIEW_ID" "$DATA_VIEW_TITLE" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base, dv_id, title = sys.argv[1:]

headers = {
    "Content-Type": "application/json",
    "kbn-xsrf": "true",
}

def request(method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

status, payload = request(
    "GET",
    f"{base}/api/data_views/data_view/{dv_id}",
)

if status == 200:
    update_body = {
        "data_view": {
            "title": title,
            "timeFieldName": "@timestamp",
            "name": title,
        },
        "refresh_fields": True,
    }

    status, payload = request(
        "POST",
        f"{base}/api/data_views/data_view/{dv_id}",
        update_body,
    )

    if status != 200:
        print(
            f"ERROR: failed to update data view ({status}): {payload[:1000]}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"    updated data view: {dv_id}")

elif status == 404:
    create_body = {
        "data_view": {
            "id": dv_id,
            "title": title,
            "timeFieldName": "@timestamp",
            "name": title,
        },
        "override": True,
    }

    status, payload = request(
        "POST",
        f"{base}/api/data_views/data_view",
        create_body,
    )

    if status != 200:
        print(
            f"ERROR: failed to create data view ({status}): {payload[:1000]}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"    created data view: {dv_id}")

else:
    print(
        f"ERROR: failed to check data view ({status}): {payload[:1000]}",
        file=sys.stderr,
    )
    sys.exit(1)

status, payload = request(
    "GET",
    f"{base}/api/data_views/data_view/{dv_id}",
)

if status != 200:
    print(
        f"ERROR: data view verification failed ({status}): {payload[:1000]}",
        file=sys.stderr,
    )
    sys.exit(1)

obj = json.loads(payload)
dv = obj.get("data_view", {})

if dv.get("id") != dv_id:
    print(
        f"ERROR: wrong data view ID returned: {dv.get('id')!r}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"    verified data view: {dv.get('id')}")
print(f"    title: {dv.get('title')}")
print(f"    time field: {dv.get('timeFieldName')}")
PY

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
    print(
        f"ERROR: import failed for {filename}:",
        json.dumps(payload, indent=2),
        file=sys.stderr,
    )
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
echo "    Debug Logs:"
echo "    ${KIBANA_URL}/app/dashboards#/view/log-levels-dashboard"
echo "    (or ${KIBANA_URL}/app/dashboards -> 'Agent Harness — Debug Logs')"
echo
echo "    Token Usage:"
echo "    ${KIBANA_URL}/app/dashboards#/view/token-usage-dashboard"
echo "    (or ${KIBANA_URL}/app/dashboards -> 'Agent Harness — Token Usage')"