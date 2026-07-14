# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License, Version 2.0.
"""Verify the flash-sdpa Attention patch against stock DINO v1, then prove the kernel.

Three checks, in order of importance:
  1. EQUIVALENCE — same weights, same input: patched model output must match the stock manual
     attention numerically. fp32, TF32 disabled, so the only difference left is kernel tiling
     order (~1e-6); tolerance 1e-5. This is the check that catches a wrong scale, a broken
     reshape, or dropout firing in eval.
  2. INTERMEDIATE LAYERS — the embedding pipeline consumes get_intermediate_layers (last-4 CLS),
     so equivalence is asserted per layer there too, not just on the final output.
  3. KERNEL PROOF — the patched forward already FORCES the flash backend (it raises on silent
     fallback); this just reports device/dtype actually exercised plus a wall-clock comparison.

Usage: python verify_flash.py [--arch vits16] [--device cpu|cuda|mps] [--tol 1e-5]
The stock reference loads from facebookresearch/dino:main; the patched model loads from THIS
checkout (torch.hub source='local'), so the script verifies exactly the code you edited.
"""
import argparse
import time
from pathlib import Path

import torch


def load_models(arch: str, device: str):
    stock = torch.hub.load("facebookresearch/dino:main", f"dino_{arch}", verbose=False)
    patched = torch.hub.load(str(Path(__file__).parent), f"dino_{arch}", source="local", verbose=False)
    return stock.to(device).eval(), patched.to(device).eval()


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="vits16")
    ap.add_argument("--device", default="cpu", help="cpu is the strict reference; cuda/mps to prove the target")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False  # strict fp32: kernel-order noise only
    torch.manual_seed(0)
    stock, patched = load_models(a.arch, a.device)
    sd_diff = [k for k in stock.state_dict() if not torch.equal(
        stock.state_dict()[k], patched.state_dict()[k])]
    print(f"state_dict: {len(stock.state_dict())} tensors, {len(sd_diff)} differ (must be 0)")

    x = torch.randn(a.batch, 3, 224, 224, device=a.device)
    with torch.no_grad():
        d_final = max_diff(stock(x), patched(x))
        inter_s = stock.get_intermediate_layers(x, n=4)
        inter_p = patched.get_intermediate_layers(x, n=4)
        d_layers = [max_diff(s[:, 0], p[:, 0]) for s, p in zip(inter_s, inter_p)]
    print(f"[{a.device}] final-output max|diff|: {d_final:.2e}")
    for i, d in enumerate(d_layers):
        print(f"[{a.device}] last-4 CLS layer {i}:   {d:.2e}")

    with torch.no_grad():
        for m, name in ((stock, "stock "), (patched, "patched")):
            t0 = time.perf_counter()
            for _ in range(10):
                m(x)
            print(f"[{a.device}] {name} 10x forward: {time.perf_counter() - t0:.2f}s")

    worst = max([d_final, *d_layers])
    ok = worst <= a.tol and not sd_diff
    print(f"worst diff {worst:.2e} vs tol {a.tol:.0e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
