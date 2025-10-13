import ray
from hydrainfer.cluster.log_server import LogServer, LogServerConfig, LatencyBreakDownMetric
from hydrainfer.utils.zmq_utils import init_zmq_recv, init_zmq_send
import time

if __name__ == '__main__':
    server = LogServer.options(name='log_server').remote(LogServerConfig())

    log_server_found: ray.actor.ActorHandle[LogServer] = ray.get_actor('log_server')
    config = ray.get(log_server_found.get_zmq_config.remote())
    print(config)
    log_server_send = init_zmq_send(config)

    log_server_send.send_pyobj(LatencyBreakDownMetric('prefill', time.perf_counter()))
    log_server_send.send_pyobj(LatencyBreakDownMetric('decode', time.perf_counter()))