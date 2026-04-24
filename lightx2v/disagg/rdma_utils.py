from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
import time


logger = logging.getLogger(__name__)


def _collect_local_ipv4_addresses() -> list[str]:
    candidates: list[str] = []

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            address = info[4][0]
            if address and not address.startswith("127.") and address not in candidates:
                candidates.append(address)
    except Exception:
        pass

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            address = sock.getsockname()[0]
            if address and not address.startswith("127.") and address not in candidates:
                candidates.append(address)
        finally:
            sock.close()
    except Exception:
        pass

    try:
        address = socket.gethostbyname(socket.gethostname())
        if address and not address.startswith("127.") and address not in candidates:
            candidates.append(address)
    except Exception:
        pass

    return candidates


def _gid_to_ipv4(gid_text: str) -> str | None:
    """Map an IPv6 GID string to IPv4 when it is IPv4-mapped."""
    text = str(gid_text).strip()
    if not text or text == "::":
        return None
    lower = text.lower()
    if lower.startswith("::ffff:"):
        return _canonical_ipv4(text[7:])
    try:
        mapped = ipaddress.ip_address(text).ipv4_mapped
        if mapped is not None:
            return str(mapped)
    except ValueError:
        pass
    return None


def _canonical_ipv4(text: str) -> str | None:
    text = str(text).strip()
    if not text:
        return None
    try:
        return str(ipaddress.IPv4Address(text))
    except Exception:
        return None


def _preferred_rdma_ipv4() -> str | None:
    """RoCE GID row to prefer when auto-picking gid_index (multi-node / multi-homing)."""
    v = _canonical_ipv4(os.getenv("RDMA_PREFERRED_IPV4", ""))
    if v:
        return v
    return _canonical_ipv4(os.getenv("MOONCAKE_LOCAL_HOSTNAME", ""))


def resolve_gid_index(ctx, port_num: int, env_var_name: str = "RDMA_GID_INDEX") -> int:
    local_ipv4s = _collect_local_ipv4_addresses()
    preferred = _preferred_rdma_ipv4()

    env_gid = os.getenv(env_var_name, "").strip()
    if env_gid:
        try:
            idx = int(env_gid)
        except ValueError:
            idx = -1
        else:
            try:
                gid_text = str(ctx.query_gid(port_num=port_num, index=idx))
            except Exception:
                gid_text = ""
            else:
                if gid_text and gid_text != "::":
                    ipv4 = _gid_to_ipv4(gid_text)
                    if ipv4 is not None and (ipv4 in local_ipv4s or ipv4 == preferred):
                        return idx

            try:
                logger.warning(
                    "Ignoring RDMA_GID_INDEX=%s because it does not map to a local IPv4 on this host (local_ipv4s=%s preferred=%s)",
                    env_gid,
                    local_ipv4s,
                    preferred,
                )
            except Exception:
                pass

    if preferred:
        for idx in range(16):
            try:
                gid_text = str(ctx.query_gid(port_num=port_num, index=idx))
            except Exception:
                continue
            if not gid_text or gid_text == "::":
                continue
            ipv4 = _gid_to_ipv4(gid_text)
            if ipv4 == preferred:
                return idx

    mapped_candidates: list[tuple[int, str]] = []
    first_non_empty_idx: int | None = None

    for idx in range(16):
        try:
            gid_text = str(ctx.query_gid(port_num=port_num, index=idx))
        except Exception:
            continue

        if not gid_text or gid_text == "::":
            continue

        if first_non_empty_idx is None:
            first_non_empty_idx = idx

        ipv4 = _gid_to_ipv4(gid_text)
        if ipv4 is not None:
            mapped_candidates.append((idx, ipv4))
            if ipv4 in local_ipv4s:
                return idx

    if mapped_candidates:
        return mapped_candidates[0][0]

    if first_non_empty_idx is not None:
        return first_non_empty_idx

    ctx.query_gid(port_num=port_num, index=0)
    return 0


def recv_json_from_stream(sock: socket.socket, timeout_sec: float = 10.0) -> dict:
    """Read one JSON object from a TCP stream (handles split packets)."""
    decoder = json.JSONDecoder()
    chunks: list[bytes] = []
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        try:
            sock.settimeout(max(0.01, deadline - time.time()))
            chunk = sock.recv(65536)
        except socket.timeout:
            continue
        if not chunk:
            break
        chunks.append(chunk)
        payload = b"".join(chunks).decode("utf-8", errors="strict")
        try:
            obj, _ = decoder.raw_decode(payload)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    msg = b"".join(chunks).decode("utf-8", errors="ignore")
    raise RuntimeError(f"Incomplete handshake JSON from peer: {msg!r}")


def rtr_ah_dest_dlid(ctx, port_num: int, remote_lid: int) -> int:
    """Destination LID for RC QP RTR when using GRH.

    RoCE (Ethernet link layer) expects dlid 0 with a valid dgid in the GRH; some
    drivers still report a non-zero port LID, and using that in AHAttr triggers
    ibv_modify_qp RTR EINVAL on many setups.
    """
    rl = int(remote_lid)
    raw = os.getenv("RDMA_RTR_DLID", "").strip().lower()
    if raw in ("0", "zero", "roce", "eth"):
        return 0
    if raw in ("peer", "remote", "ib", "infiniband"):
        return rl
    try:
        port = ctx.query_port(port_num)
        ll = int(getattr(port, "link_layer", -1))
        local_lid = int(getattr(port, "lid", -1))
        # rdma-core ibv_port_attr.link_layer: 0 unspecified, 1 InfiniBand, 2 Ethernet (RoCE).
        if ll == 2:
            return 0
        if ll == 1:
            return rl
        # Unspecified / unknown link_layer: eRDMA and some stacks omit or mis-report;
        # RoCE uses LID 0 — using a non-zero dlid here causes RTR EINVAL.
        if local_lid == 0:
            return 0
    except Exception:
        pass
    if rl == 0:
        return 0
    return rl


def rtr_path_mtu(ctx, port_num: int) -> int:
    """Use port active MTU for RTR path_mtu (avoids hard-coded 1024 vs link mismatch)."""
    try:
        port = ctx.query_port(port_num)
        return int(port.active_mtu)
    except Exception:
        import pyverbs.enums as e

        return int(e.IBV_MTU_1024)


def rtr_path_mtu_negotiated(ctx, port_num: int, peer_active_mtu: int | None) -> int:
    """path_mtu for RTR must not exceed either peer's active MTU (IB enum ordering)."""
    local = rtr_path_mtu(ctx, port_num)
    if peer_active_mtu is None:
        return local
    try:
        peer = int(peer_active_mtu)
    except (TypeError, ValueError):
        return local
    return min(local, peer)