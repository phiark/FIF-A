import torch

from fif_mvp.utils import sparse


def test_window_cache_bounded_cpu_only():
    sparse.clear_window_cache()
    max_size = sparse.MAX_WINDOW_CACHE
    # Fill more than cache size with unique keys
    for i in range(max_size + 10):
        _ = sparse.build_window_edges(
            length=i + 2, radius=1, device=torch.device("cpu")
        )
    assert len(sparse._WINDOW_CACHE) <= max_size  # type: ignore[attr-defined]
    assert len(sparse._WINDOW_CACHE_DEVICE) <= max_size  # type: ignore[attr-defined]


def test_window_cache_reuse_cpu():
    sparse.clear_window_cache()
    edges1 = sparse.build_window_edges(length=5, radius=1, device=torch.device("cpu"))
    edges2 = sparse.build_window_edges(length=5, radius=1, device=torch.device("cpu"))
    assert edges1.data_ptr() == edges2.data_ptr()
