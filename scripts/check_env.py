#!/usr/bin/env python
"""Day 1 environment smoke test. No dataset, no training."""
import json, platform, subprocess, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

results: dict = {}


def run(name, fn):
    try:
        results[name] = {"status": "PASS", "detail": fn()}
    except Exception as e:  # noqa: BLE001
        results[name] = {"status": "FAIL", "detail": f"{type(e).__name__}: {str(e)[:200]}"}
    print(f"[{results[name]['status']}] {name}: {results[name]['detail']}")


def t_device():
    assert torch.cuda.is_available(), "CUDA not available"
    p = torch.cuda.get_device_properties(0)
    return f"{p.name} cc={torch.cuda.get_device_capability(0)} vram={p.total_memory / 2**30:.2f}GiB"


def t_conv():
    torch.manual_seed(0)
    c = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    x = torch.randn(8, 3, 240, 240, device="cuda")
    y = c(x); y.sum().backward(); torch.cuda.synchronize()
    assert torch.isfinite(c.weight.grad).all()
    return f"cudnn={torch.backends.cudnn.version()} out={tuple(y.shape)}"


def t_amp():
    c = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        y = c(torch.randn(8, 3, 240, 240, device="cuda"))
    y.float().sum().backward(); torch.cuda.synchronize()
    return str(y.dtype)


def t_resnet():
    m = torchvision.models.resnet18(weights=None).cuda().eval()
    x = torch.randn(16, 3, 240, 240, device="cuda")
    with torch.no_grad():
        l1 = m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x)))))
        l2 = m.layer2(l1); l3 = m.layer3(l2)
    torch.cuda.synchronize()
    return f"layer2={tuple(l2.shape)} layer3={tuple(l3.shape)}"


def t_interp():
    y = F.interpolate(torch.randn(1, 1, 30, 30, device="cuda"), size=(240, 240), mode="bilinear", align_corners=False)
    torch.cuda.synchronize(); return str(tuple(y.shape))


def t_linalg():  # PaDiM covariance path (cuSOLVER)
    out = []
    for dt in (torch.float32, torch.float64):
        a = torch.randn(512, 448, device="cuda", dtype=dt)
        c = a.T @ a / 511 + 0.01 * torch.eye(448, device="cuda", dtype=dt)
        inv = torch.linalg.inv(c); torch.linalg.cholesky(c); torch.linalg.eigh(c)
        out.append(f"{dt}: |C@inv-I|max={(c @ inv - torch.eye(448, device='cuda', dtype=dt)).abs().max().item():.1e}")
    torch.cuda.synchronize(); return "; ".join(out)


def t_cdist():  # PatchCore NN-search path
    d = torch.cdist(torch.randn(4096, 256, device="cuda"), torch.randn(20000, 256, device="cuda"))
    v, i = d.topk(1, largest=False, dim=1); torch.cuda.synchronize()
    return f"{tuple(v.shape)}"


def t_nms():
    b = torch.rand(100, 4, device="cuda") * 100; b[:, 2:] += b[:, :2]
    k = torchvision.ops.nms(b, torch.rand(100, device="cuda"), 0.5); torch.cuda.synchronize()
    return f"kept={len(k)}"


def t_vram():
    x = torch.empty(int(6 * 2**30 / 4), device="cuda"); x.fill_(1.0); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30; del x; torch.cuda.empty_cache()
    return f"6GiB alloc ok, peak={peak:.2f}GiB"


for n, f in [("device", t_device), ("conv_fwd_bwd_fp32", t_conv), ("conv_amp_fp16", t_amp),
             ("resnet18_features_240", t_resnet), ("bilinear_interp", t_interp),
             ("linalg_cusolver_cov448", t_linalg), ("cdist_topk", t_cdist), ("tv_nms", t_nms),
             ("vram_6GiB", t_vram)]:
    run(n, f)

smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
results["_env"] = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
                   "torchvision": torchvision.__version__, "cuda_build": torch.version.cuda,
                   "driver": smi, "arch_list": torch.cuda.get_arch_list()}
Path("results").mkdir(exist_ok=True)
Path("results/day1_env.json").write_text(json.dumps(results, indent=2))
failed = [k for k, v in results.items() if v.get("status") == "FAIL"]
print("\nFAILED:", failed or "none")
sys.exit(1 if failed else 0)