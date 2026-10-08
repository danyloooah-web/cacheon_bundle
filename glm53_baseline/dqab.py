
import sys

import torch

ENABLED = True
MAX_ROWS = 8
PARTS = 4
H, K, N, R = 64, 192, 512, 64

_NATIVE: list = []
_FRAG: dict = {}
_ATTN: dict = {}
_PENDING: dict = {}
_announced: set = set()


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_DQAB {tag} {detail}", file=sys.stderr, flush=True)


def _native():
    if not _NATIVE:
        import dqab_native

        _NATIVE.append(dqab_native)
    return _NATIVE[0]


def frag(w_t: torch.Tensor) -> torch.Tensor:
    return w_t.reshape(H, N // 16, 2, 8, K // 16, 2, 4, 2).permute(0, 1, 4, 3, 6, 5, 2, 7).contiguous()


class _WKC(torch.Tensor):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.bmm and len(args) == 2 and not kwargs and isinstance(args[1], _WKC):
            owner = getattr(args[1], "_dqab_owner", None)
            a = args[0]
            if owner in _FRAG and _eligible(owner, a):
                if owner in _PENDING:
                    raise RuntimeError("dqab: the previous absorb placeholder of this attention copy was never taken")
                with torch._C.DisableTorchFunctionSubclass():
                    plain = args[1].as_subclass(torch.Tensor)
                out = torch.empty((H, a.shape[1], N), dtype=torch.bfloat16, device=a.device)
                _PENDING[owner] = (a, out, plain)
                return out
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


def _eligible(owner, a) -> bool:
    from sglang.srt.models.deepseek_common.utils import FORWARD_ABSORB_CORE_ATTENTION_BACKENDS

    attn = _ATTN[owner]
    return (ENABLED and a.dim() == 3 and a.shape[0] == H and 0 < a.shape[1] <= MAX_ROWS and a.shape[2] == K
            and a.dtype == torch.bfloat16 and a.stride(2) == 1 and a.stride(0) % 2 == 0 and a.stride(1) % 2 == 0
            and a.data_ptr() % 4 == 0 and torch.cuda.is_current_stream_capturing()
            and attn.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS)


def install(attn) -> None:
    w_kc = getattr(attn, "w_kc", None)
    mqa = getattr(attn, "attn_mqa", None)
    if (not ENABLED or mqa is None or w_kc is None or not w_kc.is_cuda or w_kc.dtype != torch.bfloat16
            or w_kc.dim() != 3 or tuple(w_kc.shape) != (H, K, N) or w_kc.stride() != (N * K, 1, K)):
        _announce("off", f"w_kc {tuple(w_kc.shape) if w_kc is not None else None} is not the bf16 [64, 192, 512] K-contiguous buffer")
        return
    key = id(mqa)
    _FRAG[key] = frag(w_kc.transpose(1, 2))
    _ATTN[key] = attn
    wrapped = w_kc.as_subclass(_WKC)
    wrapped._dqab_owner = key
    attn.w_kc = wrapped


def fragments(attn):
    return _FRAG.get(id(getattr(attn, "attn_mqa", None)))


def take(mqa, q):
    pending = _PENDING.pop(id(mqa), None)
    if pending is None:
        return None
    if q.data_ptr() != pending[1].data_ptr() or q.shape != (pending[1].shape[1], H, N):
        raise RuntimeError("dqab: the attention call's q is not the pending absorb placeholder")
    return pending


def materialize(pending) -> None:
    a, out, plain = pending
    torch.bmm(a, plain, out=out)


def decode(mqa, pending, q_rope, k_nope, k_rope, positions, cos_sin_cache, loc, kv_rows):
    q_nope = pending[0].transpose(0, 1)
    rows = q_nope.shape[0]
    q_out = torch.empty((rows, H, N + R), dtype=torch.float8_e4m3fn, device=q_nope.device)
    _native().decode(q_nope, q_rope, _FRAG[id(mqa)], k_nope, k_rope, positions, cos_sin_cache, loc, q_out, kv_rows,
                     PARTS, True)
    _announce("served", f"{rows} rows, {PARTS} CTAs a head")
    return q_out


def usable(q_rope, k_nope, k_rope, positions, cos_sin_cache, loc) -> bool:
    return (positions.dtype == torch.int64 and loc.dtype == torch.int64 and positions.is_contiguous()
            and loc.is_contiguous() and cos_sin_cache.dtype == torch.float32 and cos_sin_cache.dim() == 2
            and cos_sin_cache.stride(1) == 1 and q_rope.shape[-1] == R and k_nope.shape[-1] == N)
