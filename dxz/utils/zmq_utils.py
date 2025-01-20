import zmq
import zmq.asyncio

"""
    usage example
    zmq_url = f"tcp://127.0.0.1:40832"
    zmq_recv = init_zmq_recv(zmq_url)
    zmq_send = init_zmq_send(init_zmq_send)
    zmq_send.send_pyobj(pyobj)
    output = await zmq_recv.recv_pyobj()
"""

def init_zmq_recv(zmq_url) -> zmq.asyncio.Socket:
    context = zmq.asyncio.Context(1)
    zmq_recv = context.socket(zmq.PULL)
    zmq_recv.bind(zmq_url)
    return zmq_recv

def init_zmq_send(zmq_url) -> zmq.sugar.socket.Socket:
    context = zmq.Context(1)     
    zmq_send = context.socket(zmq.PUSH)
    zmq_send.connect(zmq_url)
    return zmq_send