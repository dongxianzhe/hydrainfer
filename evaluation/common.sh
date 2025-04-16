SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
OUR_ROOT_PATH=$(realpath "$SCRIPT_PATH/../../")
RESULT_PATH=$(echo "$SCRIPT_PATH/result/$(date +%Y%m%d_%H%M%S)")

mkdir -p $RESULT_PATH

clean_up() {
    echo "Cleaning up..."
    pgrep -f "vllm serve" >/dev/null && pgrep -f "vllm serve" | xargs kill
    pgrep -f "dxz.entrypoint.entrypoint" >/dev/null && pgrep -f "dxz.entrypoint.entrypoint" | xargs kill
}
trap clean_up EXIT

wait_api_server() {
    local ip=$1
    local port=$2
    local name="api server at $ip $port"
    local retry=0
    while ! nc -z $ip $port; do
        echo "Waiting for $name to start..."
        retry=$((retry + 1))
        if [ $retry -gt 50 ]; then
            echo "$name failed to start after 50 attempts. Exiting."
            exit 1
        fi
        sleep 5
    done
    echo "api server is running on $ip $port"
}