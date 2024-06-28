#!/usr/bin/env bash

export ENDPOINT_URL
export AUTH=${AUTH:-}

cd /app || exit

echo "Starting TabbyLoader WebUI with endpoint URL: ${ENDPOINT_URL} and auth: ${AUTH}"

# trap SIGTERM and SIGINT to kill the python process in the container
trap 'kill -TERM $PID' TERM INT

python3 /app/webui.py --listen --endpoint_url "${ENDPOINT_URL}" "${AUTH}"
