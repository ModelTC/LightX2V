from __future__ import annotations

import os
import socket


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
    if gid_text.startswith("::ffff:"):
        return gid_text.removeprefix("::ffff:")
    return None


def resolve_gid_index(ctx, port_num: int, env_var_name: str = "RDMA_GID_INDEX") -> int:
    env_gid = os.getenv(env_var_name, "").strip()
    if env_gid:
        idx = int(env_gid)
        ctx.query_gid(port_num=port_num, index=idx)
        return idx

    local_ipv4s = _collect_local_ipv4_addresses()

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