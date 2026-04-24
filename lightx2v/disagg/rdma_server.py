import json
import os
import socket
import threading

from lightx2v.disagg.rdma_base import (
    AccessFlag,
    AHAttr,
    CQ,
    GID,
    GlobalRoute,
    IBDevice,
    MR,
    PD,
    QP,
    QPAttr,
    QPCap,
    QPInitAttr,
    QPType,
    e,
    get_device_list,
    recv_json_from_stream,
    resolve_gid_index,
    rtr_ah_dest_dlid,
    rtr_path_mtu,
    rtr_path_mtu_negotiated,
)


class RDMAServer:
    def __init__(self, iface_name=None, port_num=1, buffer_size=4096):
        self.local_psn = 123456
        self._next_psn = int(self.local_psn)
        self.port_num = port_num
        self.gid_index = 0
        self.buffer_size = int(buffer_size)
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if iface_name is None:
            env_iface = os.getenv("RDMA_IFACE", "").strip()
            if env_iface:
                iface_name = env_iface
        if iface_name is None:
            devices = get_device_list()
            if not devices:
                raise RuntimeError("No RDMA device found")
            raw_name = devices[0].name
            iface_name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name

        self.ctx = IBDevice(iface_name).open()
        if self.ctx is None:
            available = []
            for dev in get_device_list():
                dev_name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
                available.append(dev_name)
            raise RuntimeError(f"Failed to open RDMA device '{iface_name}'. Available devices: {available}")

        self.pd = PD(self.ctx)
        self.cq = CQ(self.ctx, 64)
        self.gid_index = self._resolve_gid_index()

        # 创建 QP (Queue Pair)
        qp_init_attr = QPCap(max_send_wr=64, max_recv_wr=64, max_send_sge=1, max_recv_sge=1)
        qia = QPInitAttr(qp_type=QPType.RC, scq=self.cq, rcq=self.cq, cap=qp_init_attr)
        qa = QPAttr(port_num=self.port_num)
        self.qp = QP(self.pd, qia, qa)  # RC: Reliable Connected
        self._qp_init_attr = qia
        self._qp_attr = qa
        self._conn_lock = threading.Lock()
        self._active_qps = [self.qp]
        self._active_conns = []
        self._listener_socket = None

        # 关键：注册一块内存用于被远程访问
        # buffer_size 可配置，允许远程写入 (REMOTE_WRITE) 和远程读取 (REMOTE_READ)
        self.mr = MR(
            self.pd,
            self.buffer_size,
            AccessFlag.LOCAL_WRITE | AccessFlag.REMOTE_WRITE | AccessFlag.REMOTE_READ | AccessFlag.REMOTE_ATOMIC,
        )

        # 初始化缓冲区数据 (例如全为 0)
        zeros = b"\x00" * self.buffer_size
        self.mr.write(zeros, len(zeros), 0)

        mr_addr = getattr(self.mr, "addr", None)
        if mr_addr is None:
            mr_addr = self.mr.buf
        self._mr_addr = int(mr_addr)
        print(f"[Server] MR Registered. Addr: {mr_addr}, RKey: {self.mr.rkey}")

    def _resolve_gid_index(self):
        return resolve_gid_index(self.ctx, self.port_num)

    def register_memory(self, addr: int, length: int):
        """Validate a requested sub-region against server MR and return registration metadata.

        This server uses one pre-registered MR, so sub-regions are slices of that MR.
        """
        if length <= 0:
            raise ValueError("length must be positive")
        addr = int(addr)
        length = int(length)
        if addr < self._mr_addr:
            raise ValueError("addr is below MR base")
        off = addr - self._mr_addr
        if off + length > self.buffer_size:
            raise ValueError(f"region out of MR range: off={off}, length={length}, buffer_size={self.buffer_size}")
        return {
            "addr": addr,
            "length": length,
            "rkey": int(self.mr.rkey),
        }

    def read_memory(self, addr: int, length: int) -> bytes:
        """Read bytes from a validated sub-region within server MR."""
        region = self.register_memory(addr, length)
        off = int(region["addr"]) - self._mr_addr
        return self.mr.read(int(length), int(off))

    def write_memory(self, addr: int, payload: bytes):
        """Write bytes to a validated sub-region within server MR."""
        self.register_memory(addr, len(payload))
        off = int(addr) - self._mr_addr
        self.mr.write(payload, len(payload), int(off))

    def get_local_info(self, qp=None, psn=None):
        """获取本机 QP 信息，用于交换"""
        mr_addr = getattr(self.mr, "addr", None)
        if mr_addr is None:
            mr_addr = self.mr.buf
        qp = self.qp if qp is None else qp
        psn = self.local_psn if psn is None else int(psn)
        gid = self.ctx.query_gid(self.port_num, self.gid_index)
        return {
            "lid": self.ctx.query_port(self.port_num).lid,
            "qpn": qp.qp_num,
            "psn": psn,
            "gid": str(gid),
            "gid_index": self.gid_index,
            "rkey": self.mr.rkey,
            "addr": mr_addr,
            "active_mtu": int(rtr_path_mtu(self.ctx, self.port_num)),
        }

    @staticmethod
    def _safe_destroy_qp(qp):
        if qp is None:
            return
        close_fn = getattr(qp, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    def _alloc_qp_with_psn(self):
        with self._conn_lock:
            self._next_psn = (self._next_psn + 1) & 0xFFFFFF
            if self._next_psn == 0:
                self._next_psn = 1
            psn = self._next_psn
        qp = QP(self.pd, self._qp_init_attr, self._qp_attr)
        return qp, psn

    def _accept_one_client(self, listen_sock):
        conn, addr = listen_sock.accept()
        print(f"[Server] Connected to {addr}")

        qp, local_psn = self._alloc_qp_with_psn()
        try:
            # 1. 发送我的信息给 Client
            my_info = self.get_local_info(qp=qp, psn=local_psn)
            conn.sendall(json.dumps(my_info).encode())

            # 2. 接收 Client 的信息（可能分片，勿单次 recv）
            remote_info = recv_json_from_stream(conn, timeout_sec=30.0)
            print(f"[Server] Received remote info: QPN={remote_info['qpn']}")

            # 3. 修改 QP 状态到 RTS
            self._modify_qp_to_rts(qp, remote_info, local_psn)
        except BaseException:
            self._safe_destroy_qp(qp)
            try:
                conn.close()
            except Exception:
                pass
            raise

        with self._conn_lock:
            self._active_qps.append(qp)
            self._active_conns.append(conn)
        return conn

    def handshake(self, host="0.0.0.0", port=5566, serve_forever=True):
        """TCP handshake to exchange QP information.

        When serve_forever=True (default), accepts multiple clients on one port.
        Each client gets its own QP so multiple services can connect concurrently.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(16)
        self._listener_socket = sock
        print(f"[Server] Waiting for connection on {host}:{port}...")

        if not serve_forever:
            return self._accept_one_client(sock)

        while True:
            try:
                self._accept_one_client(sock)
            except Exception as exc:
                print(f"[Server] Handshake accept failed: {exc}")

    def _modify_qp_to_rts(self, qp, remote_info, local_psn):
        # Follow the standard RC flow: INIT -> RTR -> RTS.
        remote_lid = int(remote_info.get("lid", 0))
        heuristic_dlid = rtr_ah_dest_dlid(self.ctx, self.port_num, remote_lid)
        negotiated_mtu = int(rtr_path_mtu_negotiated(self.ctx, self.port_num, remote_info.get("active_mtu")))
        local_mtu = int(rtr_path_mtu(self.ctx, self.port_num))
        default_mtu = int(e.IBV_MTU_1024)

        mtu_candidates = []
        for v in (negotiated_mtu, local_mtu, default_mtu):
            if v not in mtu_candidates:
                mtu_candidates.append(v)
        dlid_candidates = []
        for v in (heuristic_dlid, 0, remote_lid):
            if v not in dlid_candidates:
                dlid_candidates.append(v)

        gr = GlobalRoute(dgid=GID(remote_info["gid"]), sgid_index=self.gid_index, hop_limit=1)
        last_exc = None
        for rd_atomic in (1, 0):
            for mtu in mtu_candidates:
                for dlid in dlid_candidates:
                    for is_global in (1, 0):
                        try:
                            init_attr = QPAttr(port_num=self.port_num)
                            init_attr.qp_access_flags = AccessFlag.LOCAL_WRITE | AccessFlag.REMOTE_WRITE | AccessFlag.REMOTE_READ | AccessFlag.REMOTE_ATOMIC
                            qp.to_init(init_attr)

                            rtr_attr = QPAttr(port_num=self.port_num)
                            rtr_attr.path_mtu = int(mtu)
                            rtr_attr.max_dest_rd_atomic = int(rd_atomic)
                            rtr_attr.min_rnr_timer = 12
                            rtr_attr.dest_qp_num = int(remote_info["qpn"])
                            rtr_attr.rq_psn = int(remote_info["psn"])
                            if is_global == 1:
                                rtr_attr.ah_attr = AHAttr(port_num=self.port_num, is_global=1, gr=gr, dlid=int(dlid))
                            else:
                                rtr_attr.ah_attr = AHAttr(port_num=self.port_num, is_global=0, dlid=int(dlid))
                            qp.to_rtr(rtr_attr)
                            last_exc = None
                            break
                        except Exception as exc:
                            last_exc = exc
                            continue
                    if last_exc is None:
                        break
                if last_exc is None:
                    break
            if last_exc is None:
                break
        if last_exc is not None:
            raise last_exc

        rts_attr = QPAttr(port_num=self.port_num)
        rts_attr.timeout = 14
        rts_attr.retry_cnt = 7
        rts_attr.rnr_retry = 7
        rts_attr.sq_psn = int(local_psn)
        rts_attr.max_rd_atomic = 1
        qp.to_rts(rts_attr)
        print("[Server] QP State changed to RTS")

    def wait_for_completion(self, timeout_ms=5000):
        """轮询 CQ 等待操作完成（如果是带响应的操作）"""
        # 单边写通常不需要接收方做额外操作，除非使用了带立即数或原子操作需要确认
        # 这里仅作演示，实际单边写完成后，服务端内存已变化
        pass

    def read_local_memory(self):
        """读取本地内存查看变化"""
        return self.mr.read(self.buffer_size, 0)


# 使用示例 (需在单独进程运行)
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="RDMA server demo")
#     parser.add_argument("--iface", default=None, help="RDMA device name, auto-detect when omitted")
#     parser.add_argument("--port-num", type=int, default=1, help="RDMA port number")
#     parser.add_argument("--buffer-size", type=int, default=4096, help="registered memory size in bytes")
#     parser.add_argument("--listen-host", default="0.0.0.0", help="TCP handshake listen host")
#     parser.add_argument("--listen-port", type=int, default=5566, help="TCP handshake listen port")
#     args = parser.parse_args()

#     srv = RDMAServer(iface_name=args.iface, port_num=args.port_num, buffer_size=args.buffer_size)
#     conn = srv.handshake(host=args.listen_host, port=args.listen_port)
#     conn.settimeout(10.0)
#     try:
#         marker = conn.recv(64)
#         print(f"[Server] Completion marker: {marker!r}")
#     except socket.timeout:
#         print("[Server] No completion marker received before timeout")
#     finally:
#         conn.close()
#     print("Data after operation:", srv.read_local_memory())
