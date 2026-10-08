
import sys

import torch

ENABLED = True
MAX_ROWS = 4
TRIGGER = 1
HIN, HOUT = 6144, 2624

_NATIVE: list = []
_FRAG: dict = {}
_announced: set = set()


def _announce(tag: str, detail: str) -> None:
    if tag not in _announced:
        _announced.add(tag)
        print(f"CACHEON_DFA {tag} {detail}", file=sys.stderr, flush=True)


def _native():
    if not _NATIVE:
        import dqab_native

        _NATIVE.append(dqab_native)
    return _NATIVE[0]


def fragments(attn):
    return _FRAG.get(id(attn))


def _fragments_of(weight: torch.Tensor) -> torch.Tensor:
    v = weight.view(HOUT // 32, 2, 2, 8, 4, HIN // 64, 2, 4, 2)
    return v.permute(0, 4, 5, 1, 3, 7, 6, 2, 8).contiguous().view(-1)


def _jit_backend(attn) -> bool:
    from sglang.kernels.ops.gemm import fused_a_gemm

    backend = fused_a_gemm.FusedAGemmBackend(getattr(attn, "fused_a_gemm_backend", "auto"))
    if backend == fused_a_gemm.FusedAGemmBackend.AUTO:
        backend = fused_a_gemm._AUTO_BACKEND
    return backend == fused_a_gemm.FusedAGemmBackend.JIT


def install(attn) -> bool:
    layer = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    weight = getattr(layer, "weight", None)
    why = None
    if not ENABLED:
        why = "disabled"
    elif weight is None or not torch.is_tensor(weight):
        why = "no fused q_a|kv_a projection"
    elif (tuple(weight.shape) != (HOUT, HIN) or weight.dtype != torch.bfloat16 or not weight.is_cuda
          or not weight.is_contiguous()):
        why = f"weight {tuple(weight.shape)} {weight.dtype}"
    elif getattr(layer, "bias", None) is not None:
        why = "the projection has a bias"
    elif not hasattr(attn, "_use_min_latency_fused_a_gemm"):
        why = "the attention has no min-latency fused-A switch"
    elif not _jit_backend(attn):
        why = "stock's fused-A backend is not the JIT kernel"
    if why is not None:
        _announce("stock", why)
        return False
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("dfa: the fragment copy is made in prepare, never under capture")
    frag = _fragments_of(weight)
    _FRAG[id(attn)] = frag
    native = _native()
    stock = attn.prepare_qkv_latent

    def prepare_qkv_latent(hidden_states, forward_batch):
        if attn._use_min_latency_fused_a_gemm is None:
            return stock(hidden_states, forward_batch)
        if (attn._use_min_latency_fused_a_gemm and torch.is_tensor(hidden_states) and hidden_states.dim() == 2
                and 1 <= hidden_states.shape[0] <= MAX_ROWS and hidden_states.shape[1] == HIN
                and hidden_states.dtype == torch.bfloat16 and hidden_states.stride(1) == 1
                and hidden_states.data_ptr() % 16 == 0 and (hidden_states.stride(0) * 2) % 16 == 0
                and not getattr(layer, "set_lora", False)):
            out = torch.empty((hidden_states.shape[0], HOUT), dtype=torch.bfloat16, device=hidden_states.device)
            native.dfa_run(hidden_states, frag, out, TRIGGER, True)
            _announce("served", f"<= {MAX_ROWS} rows through dfa (trigger {TRIGGER}, ring depth "
                                f"{native.dfa_depth(4)} / {native.dfa_depth(8)} chunks)")
            return out
        return stock(hidden_states, forward_batch)

    attn.prepare_qkv_latent = prepare_qkv_latent
    return True
