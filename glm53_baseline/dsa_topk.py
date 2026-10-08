
from functools import lru_cache

import torch

ENABLED = True
ROWS_MAX = 22
ROWS_NARROW = 30
NARROW_WIDTH = 131072


@lru_cache(maxsize=1)
def _native():
    import dsatk

    dsatk.prepare()
    return dsatk


def supported(scores, seq_lens, page_tables, out_page_indices, out_raw_indices=None) -> bool:
    rows = scores.shape[0]
    return (ENABLED
            and out_raw_indices is None
            and page_tables is not None
            and 0 < rows <= ROWS_NARROW
            and (rows <= ROWS_MAX or scores.shape[1] <= NARROW_WIDTH)
            and 0 < out_page_indices.shape[1] <= 2048
            and scores.dtype == torch.float32
            and scores.stride(1) == 1
            and scores.stride(0) % 4 == 0
            and scores.data_ptr() % 16 == 0
            and seq_lens.dtype == torch.int32
            and page_tables.dtype == torch.int32
            and page_tables.stride(1) == 1
            and out_page_indices.dtype == torch.int32
            and out_page_indices.is_contiguous())


_PATCHED: dict = {}


def _routed(stock):
    fn = _PATCHED.get(id(stock))
    if fn is None:
        def fn(scores, seq_lens, page_tables, out_page_indices, page_size, metadata, out_raw_indices=None):
            if supported(scores, seq_lens, page_tables, out_page_indices, out_raw_indices):
                _native().topk_auto(scores, seq_lens, page_tables, out_page_indices, page_size, True)
                return None
            return stock(scores, seq_lens, page_tables, out_page_indices, page_size, metadata, out_raw_indices)

        fn._cacheon_dsatk = True
        _PATCHED[id(stock)] = fn
    return fn


def scoped(call):
    def run(*args, **kwargs):
        if not ENABLED:
            return call(*args, **kwargs)
        import sglang.kernels.ops.attention.dsv4.topk as topk_module

        stock = topk_module.topk_transform_paged_v2
        if getattr(stock, "_cacheon_dsatk", False):
            return call(*args, **kwargs)
        topk_module.topk_transform_paged_v2 = _routed(stock)
        try:
            return call(*args, **kwargs)
        finally:
            topk_module.topk_transform_paged_v2 = stock

    return run
