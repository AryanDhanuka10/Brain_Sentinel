import pytest
import torch
import torchvision

# Hardware gate: these Day-1 assertions are strict wherever a CUDA device exists (the RTX 4060 laptop).
# On a CPU-only machine (CI, Docker CPU fallback) they are skipped, not weakened.
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device (CPU fallback run)")


@needs_cuda
def test_cuda_available():
    assert torch.cuda.is_available()


@needs_cuda
def test_device_is_rtx_4060():
    assert "4060" in torch.cuda.get_device_name(0)


@needs_cuda
def test_vram_headroom_for_6gib():
    free, _ = torch.cuda.mem_get_info()
    assert free > 6 * 2**30


def test_pinned_torch_pair():
    assert torch.__version__.split("+")[0] == "2.15.0"
    assert torchvision.__version__.split("+")[0] == "0.29.2"
