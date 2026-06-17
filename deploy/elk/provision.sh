#!/usr/bin/env bash
# Provision the ELK stack for Banking NLU audit observability:
#   1. install the Elasticsearch index template (mappings for nlu-audit*)
#   2. import the Kibana data view + visualizations + dashboard
#
# Usage:  bash deploy/elk/provision.sh
# Env:    ES_URL (default http://localhost:9200), KIBANA_URL (default http://localhost:5601)
set -euo pipefail

ES_URL="${ES_URL:-http://localhost:9200}"
KIBANA_URL="${KIBANA_URL:-http://localhost:5601}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Waiting for Elasticsearch at ${ES_URL} ..."
until curl -fs "${ES_URL}/_cluster/health" >/dev/null 2>&1; do sleep 3; done
echo "    Elasticsearch is up."

echo "==> Installing index template 'nlu-audit'"
curl -fsS -X PUT "${ES_URL}/_index_template/nlu-audit" \
  -H 'Content-Type: application/json' \
  --data-binary "@${HERE}/index-template.json" >/dev/null
echo "    Template installed."

echo "==> Waiting for Kibana at ${KIBANA_URL} ..."
until curl -fs "${KIBANA_URL}/api/status" >/dev/null 2>&1; do sleep 5; done
echo "    Kibana is up."

echo "==> Importing Kibana saved objects (data view, visualizations, dashboard)"
curl -fsS -X POST "${KIBANA_URL}/api/saved_objects/_import?overwrite=true" \
  -H 'kbn-xsrf: true' \
  --form file=@"${HERE}/kibana/dashboards.ndjson" >/dev/null
echo "    Saved objects imported."

echo "Done. Open Kibana: ${KIBANA_URL}/app/dashboards"
