import torch
import torch.distributed as dist

_G = 1 << 30
_llm = None


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
    vllm = " ".join(f"{t} {g:.2f}{'(asleep)' if s else '(awake)'}" for t, (g, s) in pools.items()) or "none"
    print(
        f"[mem] {tag:<30} alloc {alloc - slept:6.2f} | peak since last {peak - slept:6.2f} | "
        f"reserved {reserved - slept:6.2f} | device {device:6.2f} | non-PyTorch {device - (reserved - slept):5.2f} GiB"
        f" | vLLM {vllm}",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
