

import numpy as np
import zmq
import torch
from mooncake.engine import TransferEngine

def main():
    # Initialize ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect(f"tcp://localhost:5555")
    
    # Wait for buffer info from server
    print("Waiting for server buffer information...")
    buffer_info = socket.recv_json()
    server_session_id = buffer_info["session_id"]
    server_ptr = buffer_info["ptr"]
    server_len = buffer_info["len"]
    print(f"Received server info - Session ID: {server_session_id}")
    print(f"Server buffer address: {server_ptr}, length: {server_len}")
    
    # Initialize client engine
    HOSTNAME = "localhost" # localhost for simple demo
    METADATA_SERVER = "P2PHANDSHAKE" # [ETCD_SERVER_URL, P2PHANDSHAKE, ...]
    PROTOCOL = "rdma" # [rdma, tcp, ...]
    DEVICE_NAME = "" # auto discovery if empty

    client_engine = TransferEngine()
    client_engine.initialize(
        HOSTNAME,
        METADATA_SERVER,
        PROTOCOL,
        DEVICE_NAME
    )
    session_id = f"{HOSTNAME}:{client_engine.get_rpc_port()}"
    
    # Allocate and initialize client buffer (1MB)
    client_buffer = torch.ones(1024 * 1024, dtype=torch.uint8, device=torch.device("cuda:0"))  # Fill with ones
    client_ptr = client_buffer.data_ptr()
    client_len = client_buffer.element_size() * client_buffer.nelement()
    
    # Register memory with Mooncake
    if PROTOCOL == "rdma":
        ret_value = client_engine.register_memory(client_ptr, client_len)
    if ret_value != 0:
        print("Mooncake memory registration failed.")
        raise RuntimeError("Mooncake memory registration failed.")

    print(f"Client initialized with session ID: {session_id}")

    # Transfer data from client to server
    print("Transferring data to server...")
    for _ in range(10):
        ret = client_engine.transfer_sync_write(
            server_session_id,
            client_ptr,
            server_ptr,
            min(client_len, server_len)  # Transfer minimum of both lengths
        )
    
        if ret >= 0:
            print("Transfer successful!")
        else:
            print("Transfer failed!")
    
    # Cleanup
    if PROTOCOL == "rdma":
        ret_value = client_engine.unregister_memory(client_ptr)
        if ret_value != 0:
            print("Mooncake memory deregistration failed.")
            raise RuntimeError("Mooncake memory deregistration failed.")

    socket.close()
    context.term()

if __name__ == "__main__":
    main() 
