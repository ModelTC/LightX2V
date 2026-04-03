import json
import socket
import threading
import time

import pyverbs.enums as e
from pyverbs.addr import GID, AHAttr, GlobalRoute
from pyverbs.cq import CQ
from pyverbs.device import Context, get_device_list
from pyverbs.mr import MR
from pyverbs.pd import PD
from pyverbs.qp import QP, QPAttr, QPCap, QPInitAttr
from pyverbs.wr import SGE
from pyverbs.wr import SendWR as WR


class IBDevice:
    def __init__(self, name: str):
        self.name = name

    def open(self):
        return Context(name=self.name)


class QPType:
    RC = e.IBV_QPT_RC


class WROpcode:
    RDMA_WRITE = e.IBV_WR_RDMA_WRITE
    RDMA_READ = e.IBV_WR_RDMA_READ
    ATOMIC_FETCH_AND_ADD = e.IBV_WR_ATOMIC_FETCH_AND_ADD
    ATOMIC_CMP_AND_SWP = e.IBV_WR_ATOMIC_CMP_AND_SWP


class AccessFlag:
    LOCAL_WRITE = e.IBV_ACCESS_LOCAL_WRITE
    REMOTE_WRITE = e.IBV_ACCESS_REMOTE_WRITE
    REMOTE_READ = e.IBV_ACCESS_REMOTE_READ
    REMOTE_ATOMIC = e.IBV_ACCESS_REMOTE_ATOMIC


class RDMAClient:
    def __init__(self, iface_name=None, local_buffer_size=4096):
        self.local_psn = 654321
        self.port_num = 1
        self.gid_index = 1
        if iface_name is None:
            devices = get_device_list()
            if not devices:
                raise RuntimeError("No RDMA device found")
            raw_name = devices[0].name
            iface_name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name

        self.ctx = IBDevice(iface_name).open()
        self.pd = PD(self.ctx)
        self.cq = CQ(self.ctx, 10)

        qp_init_attr = QPCap(max_send_wr=10, max_recv_wr=10, max_send_sge=1, max_recv_sge=1)
        qia = QPInitAttr(qp_type=QPType.RC, scq=self.cq, rcq=self.cq, cap=qp_init_attr)
        qa = QPAttr(port_num=self.port_num)
        self.qp = QP(self.pd, qia, qa)

        # 客户端也需要注册内存，用于发送数据的源 (Write) 或接收数据的目标 (Read)
        self.buffer_size = int(local_buffer_size)
        if self.buffer_size <= 0:
            raise ValueError("local_buffer_size must be positive")
        self.local_mr = MR(self.pd, self.buffer_size, AccessFlag.LOCAL_WRITE)
        self._io_lock = threading.RLock()

    def _ensure_local_mr_capacity(self, required_size: int):
        required = int(required_size)
        if required <= self.buffer_size:
            return
        self.buffer_size = required
        self.local_mr = MR(self.pd, self.buffer_size, AccessFlag.LOCAL_WRITE)

    def connect_to_server(self, server_ip="127.0.0.1", port=5566):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((server_ip, port))
        except Exception:
            sock.close()
            raise

        # 1. 接收 Server 信息 (包含 rkey 和 addr)
        data = sock.recv(4096)
        self.remote_info = json.loads(data.decode())
        print(f"[Client] Got Server Info: Addr={hex(self.remote_info['addr'])}, RKey={self.remote_info['rkey']}")

        # 2. 发送我的信息给 Server
        gid = self.ctx.query_gid(self.port_num, self.gid_index)
        my_info = {
            "lid": self.ctx.query_port(self.port_num).lid,
            "qpn": self.qp.qp_num,
            "psn": self.local_psn,
            "gid": str(gid),
            "gid_index": self.gid_index,
        }
        sock.sendall(json.dumps(my_info).encode())

        # 3. 修改 QP 状态
        self._modify_qp_to_rts()
        self.sock = sock
        print("[Client] Connection established (RTS)")

    def _modify_qp_to_rts(self):
        # Follow the standard RC flow: INIT -> RTR -> RTS.
        init_attr = QPAttr(port_num=self.port_num)
        init_attr.qp_access_flags = AccessFlag.LOCAL_WRITE | AccessFlag.REMOTE_WRITE | AccessFlag.REMOTE_READ | AccessFlag.REMOTE_ATOMIC
        self.qp.to_init(init_attr)

        rtr_attr = QPAttr(port_num=self.port_num)
        rtr_attr.path_mtu = e.IBV_MTU_1024
        rtr_attr.max_dest_rd_atomic = 1
        rtr_attr.min_rnr_timer = 12
        rtr_attr.dest_qp_num = int(self.remote_info["qpn"])
        rtr_attr.rq_psn = int(self.remote_info["psn"])

        remote_lid = int(self.remote_info.get("lid", 0))
        remote_gid_index = int(self.remote_info.get("gid_index", self.gid_index))
        gr = GlobalRoute(dgid=GID(self.remote_info["gid"]), sgid_index=remote_gid_index)
        rtr_attr.ah_attr = AHAttr(port_num=self.port_num, is_global=1, gr=gr, dlid=remote_lid)
        self.qp.to_rtr(rtr_attr)

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
            old_v = int.from_bytes(old, byteorder="big", signed=False)
            return old_v

    def rdma_cas(self, remote_addr, compare_value, swap_value, rkey=None):
        """Execute true remote atomic compare-and-swap and return previous value."""
        with self._io_lock:
            self._ensure_local_mr_capacity(8)

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
            old_v = int.from_bytes(old, byteorder="big", signed=False)
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
                    raise Exception(f"WC Error: {status}, vendor_err: {vendor_err}")
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
