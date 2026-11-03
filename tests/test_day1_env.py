import torch
import torchvision


def test_cuda_available():
    assert torch.cuda.is_available()


def test_device_is_rtx_4060():
    assert "4060" in torch.cuda.get_device_name(0)


def test_vram_headroom_for_6gib():
    free, _ = torch.cuda.mem_get_info()
    assert free > 6 * 2**30


def test_pinned_torch_pair():
    assert torch.__version__.split("+")[0] == "2.15.0"
    assert torchvision.__version__.split("+")[0] == "0.29.2"
