
import numpy as np
import zmq
import torch
from mooncake.engine import TransferEngine

def main():
    # Initialize ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://*:5555")  # Bind to port 5555 for buffer info

    HOSTNAME = "localhost" # localhost for simple demo
    METADATA_SERVER = "P2PHANDSHAKE" # [ETCD_SERVER_URL, P2PHANDSHAKE, ...]
    PROTOCOL = "rdma" # [rdma, tcp, ...]
    DEVICE_NAME = "" # auto discovery if empty
    
    # Initialize server engine
    server_engine = TransferEngine()
    server_engine.initialize(
        HOSTNAME,
        METADATA_SERVER,
        PROTOCOL,
        DEVICE_NAME
    )
    session_id = f"{HOSTNAME}:{server_engine.get_rpc_port()}"
    
    # Allocate memory on server side (1MB buffer)
    server_buffer = torch.zeros(1024 * 1024, dtype=torch.uint8, device=torch.device("cuda:1"))
    server_ptr = server_buffer.data_ptr()
    server_len = server_buffer.element_size() * server_buffer.nelement()
    
    # Register memory with Mooncake
    if PROTOCOL == "rdma":
        ret_value = server_engine.register_memory(server_ptr, server_len)
        if ret_value != 0:
            print("Mooncake memory registration failed.")
            raise RuntimeError("Mooncake memory registration failed.")

    print(f"Server initialized with session ID: {session_id}")
    print(f"Server buffer address: {server_ptr}, length: {server_len}")
    
    # Send buffer info to client
    buffer_info = {
        "session_id": session_id,
        "ptr": server_ptr,
        "len": server_len
    }
    socket.send_json(buffer_info)
    print("Buffer information sent to client")
    
    # Keep server running
    try:
        while True:
            input("Press Ctrl+C to exit...")
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        # Cleanup
        if PROTOCOL == "rdma":
            ret_value = server_engine.unregister_memory(server_ptr)
            if ret_value != 0:
                print("Mooncake memory deregistration failed.")
                raise RuntimeError("Mooncake memory deregistration failed.")

        socket.close()
        context.term()

if __name__ == "__main__":
    main() 
 