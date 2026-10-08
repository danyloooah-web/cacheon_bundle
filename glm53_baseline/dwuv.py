
import sys

import torch

ENABLED = True
MAX_ROWS = 8
PUSH = True
EARLY = True
H, K, N = 64, 512, 256

_NATIVE: list = []
_EMPTY: dict = {}
_FRAG: dict = {}
_OFFER: dict = {}
_PUSHED: dict = {}
_announced: set = set()


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_DWUV {tag} {detail}", file=sys.stderr, flush=True)


def _native():
    if not _NATIVE:
        import dwuv_native

        _NATIVE.append(dwuv_native)
    return _NATIVE[0]


def frag(w_t: torch.Tensor) -> torch.Tensor:
    return w_t.reshape(H, N // 16, 2, 8, K // 16, 2, 4, 2).permute(0, 1, 4, 3, 6, 5, 2, 7).contiguous()


def _plane(t, width: int) -> bool:
    return (t.dim() == 3 and t.shape[0] == H and t.shape[2] == width and t.dtype == torch.bfloat16 and t.is_cuda
            and t.stride(2) == 1 and t.stride(0) == width and t.stride(1) % 8 == 0 and t.data_ptr() % 16 == 0)


class _WVC(torch.Tensor):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.bmm and len(args) == 2 and isinstance(args[1], _WVC) and set(kwargs) <= {"out"}:
            key = getattr(args[1], "_dwuv_owner", None)
            a, out = args[0], kwargs.get("out")
            if (ENABLED and key in _FRAG and isinstance(a, torch.Tensor) and _plane(a, K) and 0 < a.shape[1] <= MAX_ROWS
                    and (out is None or (_plane(out, N) and out.shape[1] == a.shape[1]))
                    and torch.cuda.is_current_stream_capturing()):
                rows = a.shape[1]
                res = (torch.empty((rows, H, N), dtype=torch.bfloat16, device=a.device) if out is None
                       else out.transpose(0, 1))
                offer = _OFFER.pop(key, None)
                if key in _PUSHED:
                    raise RuntimeError("dwuv: rows sent to the peers were never taken by a scatter (stale ring slot)")
                if offer is not None and offer[2].shape[0] == rows:
                    _native().run(a.transpose(0, 1), _FRAG[key], res, True, EARLY, offer[0], offer[1], offer[2])
                    _PUSHED[key] = (res.data_ptr(), rows)
                    _announce("pushed", f"{rows} rows sent to the peers from the W_UV launch")
                else:
                    _native().run(a.transpose(0, 1), _FRAG[key], res, True, EARLY, _none(a.device), 0, _none(a.device))
                _announce("served", f"{rows} rows, 79 CTAs, scatter released {'at entry' if EARLY else 'at the end'}")
                return res.transpose(0, 1) if out is None else out
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


def install(attn) -> None:
    w_vc = getattr(attn, "w_vc", None)
    if (not ENABLED or w_vc is None or not w_vc.is_cuda or w_vc.dtype != torch.bfloat16 or w_vc.dim() != 3
            or tuple(w_vc.shape) != (H, K, N)):
        _announce("off", f"w_vc {tuple(w_vc.shape) if w_vc is not None else None} is not a bf16 [64, 512, 256] tensor")
        return
    key = id(attn)
    _FRAG[key] = frag(w_vc.transpose(1, 2))
    wrapped = w_vc.as_subclass(_WVC)
    wrapped._dwuv_owner = key
    attn.w_vc = wrapped


def offer_push(attn, ring) -> None:
    key = id(attn)
    if key in _PUSHED:
        raise RuntimeError("dwuv: rows sent to the peers were never taken by a scatter (stale ring slot)")
    if ring is None:
        _OFFER.pop(key, None)
    else:
        _OFFER[key] = ring


def take_pushed(attn, x) -> bool:
    got = _PUSHED.pop(id(attn), None)
    if got is None:
        return False
    if got != (x.data_ptr(), x.shape[0]):
        raise RuntimeError("dwuv: the rows sent to the peers are not the rows the scatter was handed")
    return True


def _none(device):
    empty = _EMPTY.get(device)
    if empty is None:
        empty = _EMPTY[device] = torch.empty(0, dtype=torch.int64, device=device)
    return empty


def fragments(attn):
    return _FRAG.get(id(attn))
