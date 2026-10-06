import torch
import torch.distributed as dist

_G = 1 << 30
_llm = None



import os

def host_mem():
    # this process: resident total, anonymous, shared (pinned cudaHostAlloc memory shows up here), peak
    st = {}
    for line in open("/proc/self/status"):
        k, v = line.split(":", 1)
        if k in ("VmRSS", "RssAnon", "RssShmem", "VmHWM"):
            st[k] = int(v.split()[0]) * 1024 / _G
    out = (f"host proc {st['VmRSS']:.2f} (anon {st['RssAnon']:.2f} shm/pinned {st['RssShmem']:.2f}"
           f" peak {st['VmHWM']:.2f})")
    # whole Slurm job (all ranks + Ray + driver): what the OOM killer acts on
    cg = f"/sys/fs/cgroup/memory/slurm/uid_{os.getuid()}/job_{os.environ.get('SLURM_JOB_ID')}"
    if os.path.isdir(cg):
        rd = lambda n: int(open(f"{cg}/{n}").read()) / _G
        stat = dict(l.split() for l in open(f"{cg}/memory.stat"))
        out += (f" | job {rd('memory.usage_in_bytes'):.1f}/{rd('memory.limit_in_bytes'):.0f} GiB"
                f" peak {rd('memory.max_usage_in_bytes'):.1f}"
                f" (rss {int(stat['total_rss'])/_G:.1f} shmem {int(stat['total_shmem'])/_G:.1f}"
                f" cache {int(stat['total_cache'])/_G:.1f})")
    return out


def register_vllm(llm):
    """Call once after the vllm.LLM is built, so mark() knows which pools are asleep."""
    global _llm
    _llm = llm


def _vllm_pools():
    """{tag: (GiB, asleep)} for vLLM's cumem pools ("weights", "kv_cache")."""
    try:
        from vllm.device_allocator.cumem import CuMemAllocator
    except ImportError:
        return {}
    allocator = CuMemAllocator.instance  # read the singleton; get_instance() would create one
    if allocator is None:
        return {}
    sizes = {}
    for data in allocator.pointer_to_data.values():
        sizes[data.tag] = sizes.get(data.tag, 0) + data.handle[1]  # handle[1] = segment size in bytes
    asleep = set()
    if _llm is not None:
        executor = _llm.llm_engine.model_executor
        asleep = set(executor.sleeping_tags) if executor.is_sleeping else set()
    return {tag: (size / _G, tag in asleep) for tag, size in sizes.items()}


def mark(tag):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    pools = _vllm_pools()
    slept = sum(gib for gib, asleep in pools.values() if asleep)  # booked by PyTorch, released on the GPU
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated() / _G
    peak = torch.cuda.max_memory_allocated() / _G
    reserved = torch.cuda.memory_reserved() / _G
    device = (total - free) / _G
    CPURAM = host_mem()
    vllm = " ".join(f"{t} {g:.2f}{'(asleep)' if s else '(awake)'}" for t, (g, s) in pools.items()) or "none"
    print(
        f"[mem] {tag:<30} alloc {alloc - slept:6.2f} | peak since last {peak - slept:6.2f} | "
        f"reserved {reserved - slept:6.2f} | device {device:6.2f} | non-PyTorch {device - (reserved - slept):5.2f} GiB"
        f" | vLLM {vllm}",
        f"CPU RAM | {CPURAM}",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
