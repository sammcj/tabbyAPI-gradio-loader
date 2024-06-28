#!/usr/bin/env bash

export ENDPOINT_URL
export AUTH

cd /app || exit

echo "Starting TabbyLoader WebUI with endpoint URL: ${ENDPOINT_URL}"

# trap SIGTERM and SIGINT to kill the python process in the container
trap 'kill -TERM $PID' TERM INT

if [[ -n ${AUTH} ]] && [[ "${AUTH}" != "" ]]; then
  python3 /app/webui.py --listen --endpoint_url "${ENDPOINT_URL}" "${AUTH}"
else
  python3 /app/webui.py --listen --endpoint_url "${ENDPOINT_URL}"
fi
