// Copied and adapted from https://github.com/triple-mu/fast-ulysses
#pragma once
// Raw PTX wrappers for TMA / mbarrier / system-scope release-acquire: lifts the kernel's
// inline asm into named __device__ __forceinline__ functions. Byte-for-byte equivalent to the
// original inline asm (clobber lists preserved verbatim).
//   - mbar_* take an already-converted uint32_t smem address (no __cvta_generic_to_shared inside).
//   - tma_load_4d / tma_store_4d / async_proxy_fence keep the trailing : "memory" / ::: "memory"
//     visibility fence. Dropping it lets the compiler reorder smem accesses across the fence,
//     causing rare data corruption that small shapes won't catch.
//   - tma_commit_group / tma_wait_group / mbar_init / mbar_arrive_expect keep their original
//     no-clobber form.
//
// The TMA / mbarrier helpers below use sm90+ only PTX (cp.async.bulk.tensor, mbarrier.*expect_tx,
// mbarrier.try_wait.parity, cp.async.bulk.commit_group/wait_group, fence.proxy.async). They are
// reached only on the TMA path, which the host gates to sm90+ at runtime (resolve_config never picks TMA on sm<9).
// For multi-arch builds that include sm80, the sm80 device pass cannot assemble these features, so
// we emit __trap() stubs for __CUDA_ARCH__ < 900; they are never executed on pre-Hopper GPUs. The
// host pass (__CUDA_ARCH__ undefined) takes the real definitions but does not codegen __device__
// bodies, so the asm is harmless there.
#include <cstdint>

namespace ulysses {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)

// --- sm80 (pre-Hopper) trap stubs: the TMA path is never selected here at runtime, so every sm90+ helper
// compiles to an unreachable __trap(). One macro covers the non-template stubs. ---
#define ULYSSES_TMA_TRAP_STUB(name, ...)                                                                               \
    __device__ __forceinline__ void name(__VA_ARGS__)                                                                  \
    {                                                                                                                  \
        __trap();                                                                                                      \
    }
ULYSSES_TMA_TRAP_STUB(mbar_init, uint32_t)
ULYSSES_TMA_TRAP_STUB(mbar_arrive_expect, uint32_t, uint32_t)
ULYSSES_TMA_TRAP_STUB(mbar_wait, uint32_t, int)
ULYSSES_TMA_TRAP_STUB(tma_load_4d, uint32_t, const void*, int, int, int, int, uint32_t)
ULYSSES_TMA_TRAP_STUB(tma_store_4d, const void*, int, int, int, int, uint32_t)
ULYSSES_TMA_TRAP_STUB(tma_commit_group)
ULYSSES_TMA_TRAP_STUB(async_proxy_fence)
#undef ULYSSES_TMA_TRAP_STUB
template<int N>
__device__ __forceinline__ void tma_wait_group()
{
    __trap();
}

#else

// Init mbarrier with expected arrival count 1. Arg is an already-__cvta'd uint32_t smem address.
__device__ __forceinline__ void mbar_init(uint32_t mbar)
{
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" ::"r"(mbar));
}

// Arrive at mbarrier and declare this phase's expected transfer bytes (for TMA complete_tx).
__device__ __forceinline__ void mbar_arrive_expect(uint32_t mbar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes));
}

// Spin-wait for the mbarrier parity flip. The 'L_%=' label is made unique so the asm can be
// inlined multiple times (a fixed 'L:' would trigger a ptxas duplicate-label error).
__device__ __forceinline__ void mbar_wait(uint32_t mbar, int phase)
{
    asm volatile(
        "{\n .reg .pred p;\n L_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n @!p bra L_%=;\n }\n" ::"r"(mbar),
        "r"(phase));
}

// TMA 4D load: global -> smem, completion counted into mbar via complete_tx. smem/mbar are
// already-__cvta'd uint32_t addresses.
__device__ __forceinline__ void
tma_load_4d(uint32_t smem, const void* map, int c0, int c1, int c2, int c3, uint32_t mbar)
{
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(smem),
                 "l"(map),
                 "r"(c0),
                 "r"(c1),
                 "r"(c2),
                 "r"(c3),
                 "r"(mbar)
                 : "memory");
}

// TMA 4D store: smem -> global (bulk_group). smem is an already-__cvta'd uint32_t address.
__device__ __forceinline__ void tma_store_4d(const void* map, int c0, int c1, int c2, int c3, uint32_t smem)
{
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
                 "[%0, {%1, %2, %3, %4}], [%5];" ::"l"(map),
                 "r"(c0),
                 "r"(c1),
                 "r"(c2),
                 "r"(c3),
                 "r"(smem)
                 : "memory");
}

// Commit one bulk_group (enclosing the TMA stores issued above).
__device__ __forceinline__ void tma_commit_group()
{
    asm volatile("cp.async.bulk.commit_group;");
}

// Wait until at most N bulk_groups remain in flight (wait for all but the last N stores). N is a template
// non-type param because the PTX immediate must be a compile-time constant; the kernel (templated on STAGES)
// calls tma_wait_group<STAGES-1>() to keep the pipeline full and tma_wait_group<0>() to drain at the end.
template<int N>
__device__ __forceinline__ void tma_wait_group()
{
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N));
}

// async-proxy visibility fence: makes subsequent generic writes visible to TMA (the async proxy).
__device__ __forceinline__ void async_proxy_fence()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

#endif  // __CUDA_ARCH__ < 900

// System-scope release store / acquire load (sm70+, used by fast_barrier on all archs; not gated).
__device__ __forceinline__ void st_release_sys_u64(uint64_t* addr, uint64_t v)
{
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(addr), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const uint64_t* addr)
{
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(addr) : "memory");
    return v;
}

}  // namespace ulysses
