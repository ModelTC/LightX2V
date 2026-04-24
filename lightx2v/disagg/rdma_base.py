"""Shared pyverbs imports and thin RDMA types for client/server."""

from __future__ import annotations

import pyverbs.enums as e
from pyverbs.addr import GID, AHAttr, GlobalRoute
from pyverbs.cq import CQ
from pyverbs.device import Context, get_device_list
from pyverbs.mr import MR
from pyverbs.pd import PD
from pyverbs.qp import QP, QPAttr, QPCap, QPInitAttr
from pyverbs.wr import SGE
from pyverbs.wr import SendWR as WR

from lightx2v.disagg.rdma_utils import (
    recv_json_from_stream,
    resolve_gid_index,
    rtr_ah_dest_dlid,
    rtr_path_mtu,
    rtr_path_mtu_negotiated,
)


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


__all__ = [
    "AccessFlag",
    "AHAttr",
    "CQ",
    "Context",
    "GID",
    "GlobalRoute",
    "IBDevice",
    "MR",
    "PD",
    "QP",
    "QPAttr",
    "QPCap",
    "QPInitAttr",
    "QPType",
    "SGE",
    "WR",
    "WROpcode",
    "e",
    "get_device_list",
    "recv_json_from_stream",
    "resolve_gid_index",
    "rtr_ah_dest_dlid",
    "rtr_path_mtu",
    "rtr_path_mtu_negotiated",
]
