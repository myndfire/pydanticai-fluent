##!/usr/bin/env bash
# Provision the "Agent Harness — Log Levels" dashboard into Kibana.
#
# Creates/updates the required data view first, then imports the Lens/dashboard
# saved objects. Safe to run repeatedly.
#
# Requires: a running Kibana, curl, python3.
#
# Usage:
#   ./kibana/provision-log-levels-dashboard.sh
#   KIBANA_URL=http://localhost:5601 ./kibana/provision-log-levels-dashboard.sh

set -euo pipefail

KIBANA_URL="${KIBANA_URL:-http://localhost:5601}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NDJSON="${SCRIPT_DIR}/saved-objects/log-levels.ndjson"

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
            payload = r.read().decode()
            return r.status, payload
    except urllib.error.HTTPError as e:
        payload = e.read().decode()
        return e.code, payload


# Check whether the exact ID already exists.
status, payload = request(
    "GET",
    f"{base}/api/data_views/data_view/{dv_id}",
)

if status == 200:
    # Existing data view: update it.
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
    # Data view does not exist: create it with the exact ID referenced by Lens.
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


# Verify the exact ID is now resolvable before importing Lens objects.
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

echo "==> Importing saved objects from ${NDJSON} ..."

result="$(
  curl -fsS -X POST \
    "${KIBANA_URL}/api/saved_objects/_import?overwrite=true" \
    -H "kbn-xsrf: true" \
    --form "file=@${NDJSON}"
)"

python3 - "$result" <<'PY'
import json
import sys

r = json.loads(sys.argv[1])

if not r.get("success"):
    print("ERROR: import failed:", json.dumps(r, indent=2), file=sys.stderr)
    sys.exit(1)

n = len(r.get("successResults", []))
print(f"    imported {n} saved objects")

for sr in r.get("successResults", []):
    print(f"      - {sr['type']}: {sr['id']}")
PY

echo
echo "==> Done. Open the dashboard:"
echo "    ${KIBANA_URL}/app/dashboards#/view/log-levels-dashboard"
echo "    (or ${KIBANA_URL}/app/dashboards -> 'Agent Harness — Log Levels')"