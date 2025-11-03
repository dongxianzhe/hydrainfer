SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
OUR_ROOT_PATH=$(realpath "$SCRIPT_PATH/../../")
COMMON_SCRIPT_PATH="$SCRIPT_PATH/../"

clean_up() {
    echo "Cleaning up..."
    pgrep -f "vllm serve" >/dev/null && pgrep -f "vllm serve" | xargs kill
    pgrep -f "hydrainfer.entrypoint.entrypoint" >/dev/null && pgrep -f "hydrainfer.entrypoint.entrypoint" | xargs kill
    pgrep -f "text-generation-launcher" >/dev/null && pgrep -f "text-generation-launcher" | xargs kill
    pgrep -f "sglang.launch_server" >/dev/null && pgrep -f "sglang.launch_server" | xargs kill
    pgrep -f "lmdeploy serve" >/dev/null && pgrep -f "lmdeploy serve" | xargs kill
    # conda run -n hydrainfer ray stop
}
trap clean_up EXIT

wait_api_server() {
    local ip=$1
    local port=$2
    local pid=$3
    local name="api server at $ip $port"
    local retry=0
    while ! nc -z $ip $port; do
        echo "Waiting for $name to start..."
        retry=$((retry + 1))
        if ! kill -0 $pid 2>/dev/null; then
            echo "Backend process with PID $pid has exited."
            return 1
        fi
        if [ $retry -gt 50 ]; then
            echo "$name failed to start after 50 attempts. Exiting."
            return 1
        fi
        sleep 5
    done
    echo "api server is running on $ip $port"
    return 0
}

get_free_gpus() {
    free_gpus=$(conda run -n hydrainfer --no-capture-output python $COMMON_SCRIPT_PATH/get_free_gpus.py)
    echo "$free_gpus"
}

find_free_port() {
  local port=10000

  while true; do
    if ! lsof -i :$port &>/dev/null; then
      echo "$port"
      return 0
    fi
    ((port++))
  done
}
