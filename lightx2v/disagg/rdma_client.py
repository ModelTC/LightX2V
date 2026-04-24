import json
import logging
import os
import random
import socket
import threading
import time

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
    SGE,
    WR,
    WROpcode,
    e,
    get_device_list,
    recv_json_from_stream,
    resolve_gid_index,
    rtr_ah_dest_dlid,
    rtr_path_mtu,
    rtr_path_mtu_negotiated,
)

logger = logging.getLogger(__name__)


class RDMAClient:
    def __init__(self, iface_name=None, local_buffer_size=4096):
        self.local_psn = 654321
        self._next_psn = (int(time.time() * 1000000) & 0xFFFFFF) or 1
        self.port_num = 1
        self.gid_index = 0
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
        self.pd = PD(self.ctx)
        self.cq = CQ(self.ctx, 64)
        self.gid_index = self._resolve_gid_index()

        qp_init_attr = QPCap(max_send_wr=64, max_recv_wr=64, max_send_sge=1, max_recv_sge=1)
        self._qia = QPInitAttr(qp_type=QPType.RC, scq=self.cq, rcq=self.cq, cap=qp_init_attr)
        self._qa = QPAttr(port_num=self.port_num)
        self.qp = QP(self.pd, self._qia, self._qa)

        # 客户端也需要注册内存，用于发送数据的源 (Write) 或接收数据的目标 (Read)
        self.buffer_size = int(local_buffer_size)
        if self.buffer_size <= 0:
            raise ValueError("local_buffer_size must be positive")
        self.local_mr = MR(self.pd, self.buffer_size, AccessFlag.LOCAL_WRITE)
        self._io_lock = threading.RLock()
        self._connected_server_ip: str | None = None
        self._connected_server_port: int | None = None
        self._qp_error_state: bool = False
        self._last_wc_error_message: str = ""

    def has_qp_error(self) -> bool:
        return self._qp_error_state

    def last_wc_error_message(self) -> str:
        return self._last_wc_error_message

    def _wc_status_name(self, status: int | None) -> str:
        if status is None:
            return "UNKNOWN"
        status_map = {
            getattr(e, "IBV_WC_SUCCESS", -1): "IBV_WC_SUCCESS",
            getattr(e, "IBV_WC_LOC_LEN_ERR", -2): "IBV_WC_LOC_LEN_ERR",
            getattr(e, "IBV_WC_LOC_QP_OP_ERR", -3): "IBV_WC_LOC_QP_OP_ERR",
            getattr(e, "IBV_WC_LOC_PROT_ERR", -4): "IBV_WC_LOC_PROT_ERR",
            getattr(e, "IBV_WC_WR_FLUSH_ERR", -5): "IBV_WC_WR_FLUSH_ERR",
            getattr(e, "IBV_WC_MW_BIND_ERR", -6): "IBV_WC_MW_BIND_ERR",
            getattr(e, "IBV_WC_BAD_RESP_ERR", -7): "IBV_WC_BAD_RESP_ERR",
            getattr(e, "IBV_WC_LOC_ACCESS_ERR", -8): "IBV_WC_LOC_ACCESS_ERR",
            getattr(e, "IBV_WC_REM_INV_REQ_ERR", -9): "IBV_WC_REM_INV_REQ_ERR",
            getattr(e, "IBV_WC_REM_ACCESS_ERR", -10): "IBV_WC_REM_ACCESS_ERR",
            getattr(e, "IBV_WC_REM_OP_ERR", -11): "IBV_WC_REM_OP_ERR",
            getattr(e, "IBV_WC_RETRY_EXC_ERR", -12): "IBV_WC_RETRY_EXC_ERR",
            getattr(e, "IBV_WC_RNR_RETRY_EXC_ERR", -13): "IBV_WC_RNR_RETRY_EXC_ERR",
            getattr(e, "IBV_WC_REM_ABORT_ERR", -14): "IBV_WC_REM_ABORT_ERR",
        }
        return status_map.get(status, f"IBV_WC_STATUS_{status}")

    def _resolve_gid_index(self):
        return resolve_gid_index(self.ctx, self.port_num)

    def _alloc_local_psn(self):
        self._next_psn = (self._next_psn + 1) & 0xFFFFFF
        if self._next_psn == 0:
            self._next_psn = 1
        self.local_psn = self._next_psn
        return self.local_psn

    def _reset_qp(self):
        old_qp = getattr(self, "qp", None)
        self.qp = QP(self.pd, self._qia, self._qa)
        if old_qp is not None:
            close_fn = getattr(old_qp, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    def _ensure_local_mr_capacity(self, required_size: int):
        required = int(required_size)
        if required <= self.buffer_size:
            return
        self.buffer_size = required
        self.local_mr = MR(self.pd, self.buffer_size, AccessFlag.LOCAL_WRITE)

    def connect_to_server(self, server_ip="127.0.0.1", port=5566):
        max_retries = max(1, int(os.getenv("RDMA_CLIENT_CONNECT_RETRIES", "30")))
        connect_timeout_sec = float(os.getenv("RDMA_CLIENT_CONNECT_TIMEOUT_SEC", "2.0"))
        backoff_base_sec = float(os.getenv("RDMA_CLIENT_BACKOFF_BASE_SEC", "0.1"))
        backoff_max_sec = float(os.getenv("RDMA_CLIENT_BACKOFF_MAX_SEC", "2.0"))
        jitter_ratio = float(os.getenv("RDMA_CLIENT_BACKOFF_JITTER", "0.2"))

        last_exc = None
        for attempt in range(1, max_retries + 1):
            sock = None
            try:
                old_sock = getattr(self, "sock", None)
                if old_sock is not None:
                    try:
                        old_sock.close()
                    except Exception:
                        pass
                    self.sock = None

                self._reset_qp()
                self._alloc_local_psn()

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(connect_timeout_sec)
                sock.connect((server_ip, port))

                # 1. 接收 Server 信息 (包含 rkey 和 addr)
                remote_info = recv_json_from_stream(sock, timeout_sec=connect_timeout_sec)
                if not isinstance(remote_info, dict):
                    raise RuntimeError(f"Invalid handshake payload type: {type(remote_info)}")
                required_keys = {"addr", "rkey", "qpn", "psn", "gid"}
                missing = required_keys.difference(remote_info.keys())
                if missing:
                    raise RuntimeError(f"Handshake missing keys: {sorted(missing)}")
                self.remote_info = remote_info
                print(f"[Client] Got Server Info: Addr={hex(int(self.remote_info['addr']))}, RKey={self.remote_info['rkey']}")

                # 2. 发送我的信息给 Server
                gid = self.ctx.query_gid(port_num=self.port_num, index=self.gid_index)
                my_info = {
                    "lid": self.ctx.query_port(port_num=self.port_num).lid,
                    "qpn": self.qp.qp_num,
                    "psn": self.local_psn,
                    "gid": str(gid),
                    "gid_index": self.gid_index,
                    "active_mtu": int(rtr_path_mtu(self.ctx, self.port_num)),
                }
                sock.sendall(json.dumps(my_info).encode())

                # 3. 修改 QP 状态
                self._modify_qp_to_rts()
                sock.settimeout(None)
                self.sock = sock
                self._connected_server_ip = str(server_ip)
                self._connected_server_port = int(port)
                self._qp_error_state = False
                self._last_wc_error_message = ""
                print(f"[Client] Connection established (RTS) to {server_ip}:{port} at attempt {attempt}/{max_retries}")
                return
            except Exception as exc:
                last_exc = exc
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass

                if attempt < max_retries:
                    backoff = min(backoff_max_sec, backoff_base_sec * (2 ** (attempt - 1)))
                    if jitter_ratio > 0:
                        jitter = random.uniform(1.0 - jitter_ratio, 1.0 + jitter_ratio)
                        backoff = max(0.01, backoff * jitter)
                    print(f"[Client] Handshake attempt {attempt}/{max_retries} failed to {server_ip}:{port}: {exc}. Retrying in {backoff:.2f}s")
                    time.sleep(backoff)

        raise RuntimeError(f"RDMA client failed to connect to {server_ip}:{port} after {max_retries} attempts") from last_exc

    def _modify_qp_to_rts(self):
        # Follow the standard RC flow: INIT -> RTR -> RTS.
        remote_lid = int(self.remote_info.get("lid", 0))
        heuristic_dlid = rtr_ah_dest_dlid(self.ctx, self.port_num, remote_lid)
        negotiated_mtu = int(rtr_path_mtu_negotiated(self.ctx, self.port_num, self.remote_info.get("active_mtu")))
        local_mtu = int(rtr_path_mtu(self.ctx, self.port_num))
        default_mtu = int(e.IBV_MTU_1024)

        # Some eRDMA/RoCE stacks are strict about dlid/mtu combinations; try safe fallbacks.
        mtu_candidates = []
        for v in (negotiated_mtu, local_mtu, default_mtu):
            if v not in mtu_candidates:
                mtu_candidates.append(v)
        dlid_candidates = []
        for v in (heuristic_dlid, 0, remote_lid):
            if v not in dlid_candidates:
                dlid_candidates.append(v)

        gr = GlobalRoute(dgid=GID(self.remote_info["gid"]), sgid_index=self.gid_index, hop_limit=1)
        last_exc = None
        for rd_atomic in (1, 0):
            for mtu in mtu_candidates:
                for dlid in dlid_candidates:
                    for is_global in (1, 0):
                        try:
                            init_attr = QPAttr(port_num=self.port_num)
                            init_attr.qp_access_flags = AccessFlag.LOCAL_WRITE | AccessFlag.REMOTE_WRITE | AccessFlag.REMOTE_READ | AccessFlag.REMOTE_ATOMIC
                            self.qp.to_init(init_attr)

                            rtr_attr = QPAttr(port_num=self.port_num)
                            rtr_attr.path_mtu = int(mtu)
                            rtr_attr.max_dest_rd_atomic = int(rd_atomic)
                            rtr_attr.min_rnr_timer = 12
                            rtr_attr.dest_qp_num = int(self.remote_info["qpn"])
                            rtr_attr.rq_psn = int(self.remote_info["psn"])
                            # Some drivers require GRH(is_global=1), others only accept non-GRH.
                            if is_global == 1:
                                rtr_attr.ah_attr = AHAttr(port_num=self.port_num, is_global=1, gr=gr, dlid=int(dlid))
                            else:
                                rtr_attr.ah_attr = AHAttr(port_num=self.port_num, is_global=0, dlid=int(dlid))
                            self.qp.to_rtr(rtr_attr)
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
        rts_attr.sq_psn = self.local_psn
        rts_attr.max_rd_atomic = 1
        self.qp.to_rts(rts_attr)

    def rdma_write(self, data_bytes, notify_server: bool = False):
        """执行单边写：将本地数据直接写入远程内存"""
        self._ensure_local_mr_capacity(len(data_bytes))

        # 1. 准备本地数据
        padded = data_bytes.ljust(self.buffer_size, b"\x00")
        self.local_mr.write(padded, len(padded), 0)

        # 2. 构造 WR (Work Request)
        sge = SGE(self.local_mr.buf, len(data_bytes), self.local_mr.lkey)
        wr = WR(
            wr_id=123,
            opcode=WROpcode.RDMA_WRITE,
            num_sge=1,
            sg=[sge],
            send_flags=e.IBV_SEND_SIGNALED,
        )
        wr.set_wr_rdma(int(self.remote_info["rkey"]), int(self.remote_info["addr"]))

        # 3. 提交请求
        self.qp.post_send(wr)

        # 4. 轮询完成队列 (如果之前设置了 SIGNALED)
        # 对于纯单边写，如果不要求确认，可以不用轮询，这就是"零拷贝零中断"的精髓
        # 但为了演示成功，我们这里简单轮询一下
        self._poll_cq()
        # Optional demo-path notification channel; rdma_buffer path does not rely on it.
        if notify_server and hasattr(self, "sock") and self.sock is not None:
            try:
                self.sock.sendall(b"WRITE_DONE")
            except (BrokenPipeError, OSError):
                self.sock = None

    def rdma_read(self, length):
        """执行单边读：直接从远程内存读取数据到本地"""
        self._ensure_local_mr_capacity(length)
        sge = SGE(self.local_mr.buf, length, self.local_mr.lkey)
        wr = WR(
            wr_id=124,
            opcode=WROpcode.RDMA_READ,
            num_sge=1,
            sg=[sge],
            send_flags=e.IBV_SEND_SIGNALED,
        )
        wr.set_wr_rdma(int(self.remote_info["rkey"]), int(self.remote_info["addr"]))

        self.qp.post_send(wr)

        self._poll_cq()
        return self.local_mr.read(length, 0)

    def rdma_write_to(self, remote_addr, data_bytes, rkey=None):
        """Write bytes to an explicit remote address.

        Keeps compatibility with existing rdma_write implementation by temporarily
        overriding remote_info addr/rkey for this operation.
        """
        with self._io_lock:
            old_addr = self.remote_info["addr"]
            old_rkey = self.remote_info["rkey"]
            self.remote_info["addr"] = int(remote_addr)
            if rkey is not None:
                self.remote_info["rkey"] = int(rkey)
            try:
                self.rdma_write(data_bytes, notify_server=False)
            except Exception as exc:
                raise RuntimeError(
                    f"rdma_write_to failed server={self._connected_server_ip}:{self._connected_server_port} remote_addr={int(remote_addr)} length={len(data_bytes)} rkey={self.remote_info.get('rkey')}"
                ) from exc
            finally:
                self.remote_info["addr"] = old_addr
                self.remote_info["rkey"] = old_rkey

    def rdma_read_from(self, remote_addr, length, rkey=None):
        """Read bytes from an explicit remote address."""
        with self._io_lock:
            old_addr = self.remote_info["addr"]
            old_rkey = self.remote_info["rkey"]
            self.remote_info["addr"] = int(remote_addr)
            if rkey is not None:
                self.remote_info["rkey"] = int(rkey)
            try:
                return self.rdma_read(int(length))
            except Exception as exc:
                raise RuntimeError(
                    f"rdma_read_from failed server={self._connected_server_ip}:{self._connected_server_port} remote_addr={int(remote_addr)} length={int(length)} rkey={self.remote_info.get('rkey')}"
                ) from exc
            finally:
                self.remote_info["addr"] = old_addr
                self.remote_info["rkey"] = old_rkey

    def rdma_faa(self, remote_addr, add_value, rkey=None):
        """Execute true remote atomic fetch-and-add and return previous value."""
        with self._io_lock:
            self._ensure_local_mr_capacity(8)

            # The original remote value will be written into this local buffer.
            self.local_mr.write(b"\x00" * 8, 8, 0)

            sge = SGE(self.local_mr.buf, 8, self.local_mr.lkey)
            wr = WR(
                wr_id=125,
                opcode=WROpcode.ATOMIC_FETCH_AND_ADD,
                num_sge=1,
                sg=[sge],
                send_flags=e.IBV_SEND_SIGNALED,
            )

            target_rkey = int(self.remote_info["rkey"] if rkey is None else rkey)
            add_u64 = int(add_value) & ((1 << 64) - 1)
            wr.set_wr_atomic(target_rkey, int(remote_addr), add_u64, 0)

            self.qp.post_send(wr)
            self._poll_cq()

            old = self.local_mr.read(8, 0)
            old_v = int.from_bytes(old, byteorder="little", signed=False)
            return old_v

    def rdma_cas(self, remote_addr, compare_value, swap_value, rkey=None):
        """Execute true remote atomic compare-and-swap and return previous value."""
        with self._io_lock:
            self._ensure_local_mr_capacity(8)

            # The original remote value will be written into this local buffer.
            self.local_mr.write(b"\x00" * 8, 8, 0)

            sge = SGE(self.local_mr.buf, 8, self.local_mr.lkey)
            wr = WR(
                wr_id=126,
                opcode=WROpcode.ATOMIC_CMP_AND_SWP,
                num_sge=1,
                sg=[sge],
                send_flags=e.IBV_SEND_SIGNALED,
            )

            target_rkey = int(self.remote_info["rkey"] if rkey is None else rkey)
            compare_u64 = int(compare_value) & ((1 << 64) - 1)
            swap_u64 = int(swap_value) & ((1 << 64) - 1)
            wr.set_wr_atomic(target_rkey, int(remote_addr), compare_u64, swap_u64)

            self.qp.post_send(wr)
            self._poll_cq()

            old = self.local_mr.read(8, 0)
            old_v = int.from_bytes(old, byteorder="little", signed=False)
            return old_v

    def _poll_cq(self):
        """简单的轮询"""
        while True:
            poll_ret = self.cq.poll(1)
            if not isinstance(poll_ret, tuple) or len(poll_ret) != 2:
                raise RuntimeError(f"Unexpected CQ poll return: {poll_ret}")
            num_wc, wc_list = poll_ret
            if num_wc > 0 and wc_list:
                wc = wc_list[0]
                status = getattr(wc, "status", None)
                if status is None:
                    raise RuntimeError(f"Unexpected WC object: {wc}")
                if status != e.IBV_WC_SUCCESS:
                    vendor_err = getattr(wc, "vendor_err", None)
                    wr_id = getattr(wc, "wr_id", None)
                    opcode = getattr(wc, "opcode", None)
                    status_name = self._wc_status_name(status)
                    self._qp_error_state = True
                    self._last_wc_error_message = (
                        f"status={status}({status_name}) vendor_err={vendor_err} wr_id={wr_id} opcode={opcode} server={self._connected_server_ip}:{self._connected_server_port}"
                    )
                    logger.error(
                        "RDMA CQ failure: status=%s(%s) vendor_err=%s wr_id=%s opcode=%s server=%s:%s",
                        status,
                        status_name,
                        vendor_err,
                        wr_id,
                        opcode,
                        self._connected_server_ip,
                        self._connected_server_port,
                    )
                    raise RuntimeError(f"WC Error: {status}({status_name}), vendor_err: {vendor_err}, wr_id: {wr_id}, opcode: {opcode}")
                break
            time.sleep(0.0001)


# 使用示例
# if __name__ == "__main__":
#     cli = RDMAClient()
#     cli.connect_to_server('127.0.0.1') # 替换为服务器 IP

#     # 执行单边写
#     # msg = b"Hello RDMA!"
#     # cli.rdma_write(msg)
#     # print("Write done.")

#     # # 执行单边读
#     # data = cli.rdma_read(len(msg))
#     # print("Read data:", data)

#     # 执行单边写（rdma_write 需要 bytes-like 数据）
#     value = 123
#     payload = int(value).to_bytes(8, byteorder="little", signed=False)
#     cli.rdma_write(payload)
#     print(f"Write done. value={value}")

#     # 执行单边读
#     data = cli.rdma_read(8)
#     read_value = int.from_bytes(data, byteorder="little", signed=False)
#     print(f"Read data: raw={data} parsed={read_value}")

#     old_value = cli.rdma_faa(remote_addr=cli.remote_info["addr"], add_value=10)
#     print(f"FAA old value: {old_value}")

#     data = cli.rdma_read(8)
#     faa_read_value = int.from_bytes(data, byteorder="little", signed=False)
#     print(f"Read data after FAA: raw={data} parsed={faa_read_value}")
