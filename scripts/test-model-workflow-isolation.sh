#!/usr/bin/env bash
set -euo pipefail

# Regression test for the error examples. It verifies that model attribution
# and generic workflow scope belong to the process that emitted the record.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="${ROOT_DIR}/agent_harness_examples"
OUTPUT_DIR="$(mktemp -d)"
EXAMPLE_TIMEOUT_SECONDS="${EXAMPLE_TIMEOUT_SECONDS:-90}"
FAIL_ON_EXAMPLE_EXIT="${FAIL_ON_EXAMPLE_EXIT:-false}"
trap 'rm -rf "${OUTPUT_DIR}"' EXIT

run_example() {
    local name="$1"
    local file="$2"
    echo "Running ${name}"
    (
        cd "${EXAMPLES_DIR}"
        ERROR_HANDLING_MODEL_NAME="${ERROR_HANDLING_MODEL_NAME:-phi4-mini}" \
            HARNESS_TELEMETRY_CONSOLE=true \
            uv run "${file}"
    ) >"${OUTPUT_DIR}/${name}.log" 2>&1 &
    local pid=$!
    local deadline=$((SECONDS + EXAMPLE_TIMEOUT_SECONDS))
    while kill -0 "${pid}" 2>/dev/null; do
        if (( SECONDS >= deadline )); then
            pkill -TERM -P "${pid}" 2>/dev/null || true
            pkill -TERM -f "agent_harness_examples/${file}" 2>/dev/null || true
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 2
            pkill -KILL -P "${pid}" 2>/dev/null || true
            pkill -KILL -f "agent_harness_examples/${file}" 2>/dev/null || true
            kill -KILL "${pid}" 2>/dev/null || true
            echo "${name} exceeded ${EXAMPLE_TIMEOUT_SECONDS}s" >&2
            return 124
        fi
        sleep 1
    done
    if ! wait "${pid}"; then
        echo "${name} exited non-zero; validating captured telemetry" >&2
        if [[ "${FAIL_ON_EXAMPLE_EXIT}" == "true" ]]; then
            return 1
        fi
    fi
}

assert_contains() {
    local file="$1"
    local text="$2"
    if ! grep -Fq -- "${text}" "${file}"; then
        echo "Expected '${text}' in ${file}" >&2
        return 1
    fi
}

assert_absent() {
    local file="$1"
    local text="$2"
    if grep -Fq -- "${text}" "${file}"; then
        echo "Did not expect '${text}' in ${file}" >&2
        return 1
    fi
}

run_example "tool-errors" "02-error_handling/04_tool_errors.py"
run_example "custom-recovery" "02-error_handling/03_custom_recovery.py"

assert_contains "${OUTPUT_DIR}/tool-errors.log" 'workflow.name'
assert_contains "${OUTPUT_DIR}/tool-errors.log" 'error-handling'
assert_contains "${OUTPUT_DIR}/tool-errors.log" 'workflow.step'
assert_contains "${OUTPUT_DIR}/tool-errors.log" 'tool-errors'
assert_contains "${OUTPUT_DIR}/tool-errors.log" 'model.requested.name'
assert_contains "${OUTPUT_DIR}/tool-errors.log" 'gpt-oss:20b'
assert_absent "${OUTPUT_DIR}/tool-errors.log" 'stack-trace-test-model'

assert_contains "${OUTPUT_DIR}/custom-recovery.log" 'workflow.name'
assert_contains "${OUTPUT_DIR}/custom-recovery.log" 'error-handling'
assert_contains "${OUTPUT_DIR}/custom-recovery.log" 'workflow.step'
assert_contains "${OUTPUT_DIR}/custom-recovery.log" 'stack-trace-capture'
assert_contains "${OUTPUT_DIR}/custom-recovery.log" 'model.requested.name'
assert_contains "${OUTPUT_DIR}/custom-recovery.log" 'stack-trace-test-model'

echo "Model/workflow isolation checks passed."
