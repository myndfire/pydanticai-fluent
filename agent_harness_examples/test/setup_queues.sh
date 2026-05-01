#!/bin/bash

# Setup RabbitMQ exchanges and queues

HOST="${RABBITMQ_HOST:-localhost}"
PORT="${RABBITMQ_PORT:-15672}"
USER="${RABBITMQ_USER:-guest}"
PASS="${RABBITMQ_PASSWORD:-guest}"

echo "Creating exchanges and queues..."

# Create document exchange
curl -s -u "$USER:$PASS" -X PUT "http://$HOST:$PORT/api/exchanges/%2F/document_workflow_exchange" \
  -H "content-type:application/json" \
  -d '{"type": "direct", "durable": true}'
echo " - document_workflow_exchange"


# Create dead_letter_exchange exchange
curl -s -u "$USER:$PASS" -X PUT "http://$HOST:$PORT/api/exchanges/%2F/dead_letter_exchange" \
  -H "content-type:application/json" \
  -d '{"type": "direct", "durable": true}'
echo " - dead_letter_exchange"


# Create Queues
# Create document_input_queue
curl -s -u "$USER:$PASS" -X PUT "http://$HOST:$PORT/api/queues/%2F/document_input_queue" \
  -H "content-type:application/json" \
  -d '{"durable": true}'
echo " - document_input_queue"

# Create classified_document_queue
curl -s -u "$USER:$PASS" -X PUT "http://$HOST:$PORT/api/queues/%2F/classified_document_queue" \
  -H "content-type:application/json" \
  -d '{"durable": true}'
echo " - classified_document_queue"

# Create health_document_queue
curl -s -u "$USER:$PASS" -X PUT "http://$HOST:$PORT/api/queues/%2F/classified_document_queue" \
  -H "content-type:application/json" \
  -d '{"durable": true}'
echo " - health_document_queue"

# Create invoice_document_queue
curl -s -u "$USER:$PASS" -X PUT "http://$HOST:$PORT/api/queues/%2F/classified_document_queue" \
  -H "content-type:application/json" \
  -d '{"durable": true}'
echo " - invoice_document_queue"

# Create resume_documents_queue
curl -s -u "$USER:$PASS" -X PUT "http://$HOST:$PORT/api/queues/%2F/classified_document_queue" \
  -H "content-type:application/json" \
  -d '{"durable": true}'
echo " - resume_document_queue"

# Create dead_letter_queue
curl -s -u "$USER:$PASS" -X PUT "http://$HOST:$PORT/api/queues/%2F/dead_letter_queue" \
  -H "content-type:application/json" \
  -d '{"durable": true}'
echo " - dead_letter_queue"

# Bind document_input_queue to document_workflow_exchange
curl -s -u "$USER:$PASS" -X POST "http://$HOST:$PORT/api/bindings/%2F/e/document_workflow_exchange/q/document_input_queue" \
  -H "content-type:application/json" \
  -d '{"routing_key": "document_input_queue"}'
echo " - bound document_input_queue to document_workflow_exchange"

# Bind classified_document_queue to document_workflow_exchange
curl -s -u "$USER:$PASS" -X POST "http://$HOST:$PORT/api/bindings/%2F/e/document_workflow_exchange/q/classified_document_queue" \
  -H "content-type:application/json" \
  -d '{"routing_key": "classified_document_queue"}'
echo " - bound classified_document_queue to document_workflow_exchange"

# Bind health_document_queue to document_workflow_exchange
curl -s -u "$USER:$PASS" -X POST "http://$HOST:$PORT/api/bindings/%2F/e/document_workflow_exchange/q/classified_document_queue" \
  -H "content-type:application/json" \
  -d '{"routing_key": "health_document_queue"}'
echo " - bound health_document_queue to document_workflow_exchange"

# Bind invoice_document_queue to document_workflow_exchange
curl -s -u "$USER:$PASS" -X POST "http://$HOST:$PORT/api/bindings/%2F/e/document_workflow_exchange/q/classified_document_queue" \
  -H "content-type:application/json" \
  -d '{"routing_key": "invoice_document_queue"}'
echo " - bound invoice_document_queue to document_workflow_exchange"

# Bind resume_document_queue to document_workflow_exchange
curl -s -u "$USER:$PASS" -X POST "http://$HOST:$PORT/api/bindings/%2F/e/document_workflow_exchange/q/classified_document_queue" \
  -H "content-type:application/json" \
  -d '{"routing_key": "resume_document_queue"}'
echo " - bound resume_document_queue to document_workflow_exchange"

# Bind dead_letter_queue to dead_letter_exchange
curl -s -u "$USER:$PASS" -X POST "http://$HOST:$PORT/api/bindings/%2F/e/dead_letter_exchange/q/dead_letter_queue" \
  -H "content-type:application/json" \
  -d '{"routing_key": "dead_letter_queue"}'
echo " - bound dead_letter_queue to dead_letter_exchange"

echo "Done!"