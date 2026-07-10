// Copied and adapted from https://github.com/triple-mu/fast-ulysses
#pragma once
#include "ulysses_common.cuh"
#include <algorithm>
#include <tuple>

namespace ulysses {

// A2A launch config: TMA path uses tile_n/tile_s (stages/bdiv are fixed at 4 in all_to_all_tma.cu);
// non-TMA path uses threads/unroll/blocks.
struct A2AConfig {
    int tile_n = 0, tile_s = 1;                 // TMA path
    int threads = 512, unroll = 4, blocks = 0;  // non-TMA path
};

// Defaults per mode (near-optimal for DiT):
// mode0 -> tile_n=max(1,n_local-1), tile_s=1 (non-divisor small tile_n yields more n-tiles, higher TMA
// concurrency; safe in mode0 -- the dst n-dim is n_local, so the trailing tile clips). mode1 -> tile_n=n_local,
// tile_s=2 (mode1 needs tile_n | n_local, so a whole-block tile is the safe default).
inline A2AConfig default_config(int mode, int n_local)
{
    A2AConfig c;
    if (mode == 0) {
        c.tile_n = std::max(1, n_local - 1);
        c.tile_s = 1;
    }
    else {
        c.tile_n = std::max(1, n_local);
        c.tile_s = 2;
    }
    c.threads = 512;
    c.unroll  = 4;
    c.blocks  = 0;
    return c;
}

// Autotune/cache key (excludes b and elem -- 2B path only): ws, mode, tma(0/1),
// n_local, s_local, d. The tma bit separates TMA / non-TMA configs.
using ConfigKey = std::tuple<int, int, int, int, int, int>;

inline ConfigKey config_key(int ws, int mode, bool tma, const Ulysses4DDims& dims)
{
    return ConfigKey{ws, mode, tma ? 1 : 0, dims.n_local, dims.s_local, dims.d};
}

}  // namespace ulysses
