#!/bin/bash

# Post a message to RabbitMQ queue using Management HTTP API

# Default values
HOST="${RABBITMQ_HOST:-localhost}"
PORT="${RABBITMQ_PORT:-15672}"
USER="${RABBITMQ_USER:-guest}"
PASS="${RABBITMQ_PASSWORD:-guest}"
QUEUE="${QUEUE:-document_input_queue}"
EXCHANGE="${EXCHANGE:-document_workflow_exchange}"

# Get filename from argument or use default
FILENAME="${1:-labs.md}"

# Create JSON payload
PAYLOAD=$(echo "{\"filename\": \"$FILENAME\", \"session_id\": \"test_$(date +%s)\", \"tenant_id\": \"tenant_001\", \"user_id\": \"user_001\", \"transaction_id\": \"txn_$(date +%s)\"}" | base64 -w0)

# Post message to queue via exchange
curl -s -u "$USER:$PASS" \
  -H "content-type:application/json" \
  -X POST "http://$HOST:$PORT/api/exchanges/%2F/$EXCHANGE/publish" \
  -d "{\"properties\": {}, \"routing_key\": \"$QUEUE\", \"payload\": \"$PAYLOAD\", \"payload_encoding\": \"base64\"}"

echo ""
echo "Posted $FILENAME to queue: $QUEUE via exchange: $EXCHANGE"