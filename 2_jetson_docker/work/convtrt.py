#!/usr/bin/env python3
# ============================================================
# convtrt.py - ONNX -> TensorRT converter for Jetson Orin Nano
#                     (PatchCore / PaDiM / EfficientAD / VAE)
# ============================================================
'''
    Typical usage
    -------------
    PatchCore  (FP16, opset 17, DLA-core 0, 4 GiB workspace):

    python convtrt.py \
    --model        <model> \
    --weights_pth  models/<model_name>/model.pth \
    --meta_json    models/<model_name>/meta.json \
    --memory_bank  models/<model_name>/memory_bank.npz \ # PatchCore Only
    --stats        models/<model_name>/stats.npz \       # PaDiM Only
    --opset        17 \                                  # or 18 (JetPack6.2)
    --fp16 \
    --workspace    4096 \
    --use_dla      0 \
    --allow_gpu_fallback \
    --save_timing  models/model_name/cache.bin \
    --out_dir      models/model_name/engine
'''
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Extra
import argparse, json, shutil, subprocess, sys, textwrap, torch, numpy as np

# ---------- meta.json (only a subset is used here) ----------
class Meta(BaseModel, extra=Extra.allow):
    image_size : int
    weights_pth: str
    memory_bank: str | None = None   # PatchCore only
    stats: str | None = None         # PaDiM only

# ---------- Anomalib / VAE models ---------------------------
from anomalib.models import Patchcore, Padim, EfficientAd
from model_class.vae import ConvVAE

REGISTRY = {"patchcore":Patchcore,"padim":Padim,"efficientad":EfficientAd,"vae":ConvVAE}
KW_OK = {
    "patchcore":{"backbone","layers","coreset_ratio"},
    "padim":{"backbone","layers","n_features"},
    "efficientad":{"model_size"},
    "vae":set()
}

def build_model(name:str, meta:Meta):
    if name=="vae":
        return ConvVAE(meta.latent_dim, meta.learning_rate, meta.image_size)
    kw={k:meta.__dict__[k] for k in KW_OK[name] if k in meta.__dict__}
    if name=="patchcore" and "coreset_ratio" in kw:
        kw["coreset_sampling_ratio"]=kw.pop("coreset_ratio")
    return REGISTRY[name](**kw)

# ---------- CLI ---------------------------------------------
ap=argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""
  Typical:
    python convtrt.py --model patchcore --meta_json models/PatchCore/meta.json \\
                      --weights_pth models/PatchCore/retrained/model.pth \\
                      --opset 18 --fp16 --workspace 4096 --out_dir engine
"""))
ap.add_argument("--model", required=True, choices=list(REGISTRY))
ap.add_argument("--meta_json", required=True, type=Path)
ap.add_argument("--weights_pth", type=Path, help="override checkpoint path")
ap.add_argument("--memory_bank", type=Path, help="override memory_bank path")
ap.add_argument("--with_memory_bank", action="store_true",
                help="embed PatchCore memory_bank into ONNX")
ap.add_argument("--stats", type=Path, help="override stats path")
ap.add_argument("--with_stats", action="store_true",
                help="embed PaDiM stats(mu,inv_cov) into ONNX")
# ONNX
ap.add_argument("--opset", type=int, default=18)
# TRT flags
ap.add_argument("--fp16", action="store_true")
ap.add_argument("--int8", action="store_true")
ap.add_argument("--calib", type=Path, help="INT8 calibration cache")
ap.add_argument("--workspace", type=int, default=2048)
ap.add_argument("--use_dla", type=int)
ap.add_argument("--allow_gpu_fallback", action="store_true")
ap.add_argument("--skip_inference", action="store_true",
                help="build engine only (`--buildOnly`)")
# timing cache
ap.add_argument("--save_timing", type=Path)
ap.add_argument("--load_timing", type=Path)
ap.add_argument("--out_dir", type=Path, default=Path("runs/convert"))
args = ap.parse_args()

# ---------- resolve paths -----------------------------------
meta = Meta(**json.loads(args.meta_json.read_text()))
root = args.meta_json.parent
ckpt = (args.weights_pth or root/meta.weights_pth).resolve()
bank = (args.memory_bank or (root/meta.memory_bank) if meta.memory_bank else None)
stats = (args.stats or (root/meta.stats) if meta.stats else None)

if not ckpt.exists():
    sys.exit(f"[ERROR] checkpoint not found: {ckpt}")

# ---------- build & load ------------------------------------
model = build_model(args.model, meta)
model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)

if args.model=="patchcore" and bank and args.with_memory_bank:
    tgt = model.model if hasattr(model,"model") else model
    for fn in ("load_memory_bank","load_memory"):
        if hasattr(tgt,fn):
            getattr(tgt,fn)(bank)
            break

if args.model=="padim" and stats and args.with_stats:
    g = getattr(model.model, "gaussian", None)
    if g is not None:
        s = np.load(stats)
        g.register_buffer("mean",           torch.tensor(s["mu"]))
        g.register_buffer("inv_covariance", torch.tensor(s["cov_inv"]))
        g.is_fitted = True     # Anomalib >= .10
    else:
        print("[WARN] PaDiM gaussian not found - stats not embedded", file=sys.stderr)

model.eval()

# ---------- select network for export ----------------------
if args.model == "padim":
    import torch.nn as nn

    class PadimWithStats(nn.Module):
        def __init__(self, base: Padim):
            super().__init__()
            if hasattr(base, "feature_extractor"):
                self.fe = base.feature_extractor
            elif hasattr(base, "model") and hasattr(base.model, "feature_extractor"):
                self.fe = base.model.feature_extractor
            else:
                raise RuntimeError("Padim feature_extractor not found")
            self.register_buffer("mu",      base.model.gaussian.mean)
            self.register_buffer("inv_cov", base.model.gaussian.inv_covariance)

        def forward(self, x):
            feat_map = self.fe(x)
            if isinstance(feat_map, dict):
                feat_map = next(iter(feat_map.values()))
            B, C, H, W = feat_map.shape
            P = H * W
            feat = feat_map.reshape(B, C, P)
            diff = feat - self.mu.unsqueeze(0)
            diff_p = diff.permute(0, 2, 1)
            score = torch.einsum("bpc,pce,bpe->bp", diff_p, self.inv_cov, diff_p)
            raw, _ = score.max(dim=1)
            return raw.unsqueeze(1)

    net = PadimWithStats(model)
    out_name = "raw_score"
else:
    net = model.model if hasattr(model,"model") else model
    out_name = "output"

# ---------- ONNX export -------------------------------------
args.out_dir.mkdir(parents=True, exist_ok=True)
onnx_p = args.out_dir/"model.onnx"
dummy = torch.randn(1,3,meta.image_size,meta.image_size)
torch.onnx.export(
    net, dummy, onnx_p.as_posix(),
    opset_version=args.opset,
    input_names=["input"], output_names=[out_name],
    dynamic_axes={"input":{0:"N"}, out_name:{0:"N"}}
)
print("ONNX saved :", onnx_p)

# ---------- copy memory_bank for PatchCore ------------------
if args.model=="patchcore" and bank:
    dst = args.out_dir/bank.name
    shutil.copy(bank, dst)
    print("memory_bank copied :", dst)

# ---------- copy stats.npz for PaDiM -----------------------
if args.model=="padim" and stats:
    dst = args.out_dir/stats.name
    shutil.copy(stats, dst)
    print("stats.npz copied :", dst)

# ---------- trtexec command ---------------------------------
trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
if not Path(trtexec).exists():
    sys.exit("trtexec not found - install TensorRT CLI or mount it in container.")

plan_p = args.out_dir/"model.plan"
cmd = [
    trtexec,
    f"--onnx={onnx_p}",
    f"--saveEngine={plan_p}",
    f"--memPoolSize=workspace:{args.workspace}"
]

# precision flags
if args.fp16: cmd += ["--fp16"]
if args.int8:
    cmd += ["--int8"]
    if args.calib:
        cmd += [f"--calib={args.calib}"]
    else:
        print("[WARN] --int8 specified without --calib ; TensorRT will try "
              "dynamic INT8 calibration and often crashes. Recommend adding "
              "--calib=<cache> or re-run with --fp16 only.", file=sys.stderr)
# DLA
if args.use_dla is not None:
    cmd += [f"--useDLACore={args.use_dla}"]
    if args.allow_gpu_fallback:
        cmd += ["--allowGPUFallback"]

# build-only / skip inference
if args.skip_inference:
    cmd += ["--skipInference"]

# timing cache
if args.save_timing or args.load_timing:
    cache = args.save_timing or args.load_timing
    cmd += [f"--timingCacheFile={cache}"]

print("\n[trtexec]", " ".join(cmd), "\n")

# ---------- run ---------------------------------------------
try:
    subprocess.run(cmd, check=True, text=True)
except subprocess.CalledProcessError as e:
    print("\n[TRT ERROR] trtexec failed - full log above. "
          "Common fixes:\n"
          "  - export with lower opset (14)\n"
          "  - remove --int8 or supply --calib\n"
          "  - build on GPU first, then try DLA with --allowGPUFallback\n",
          file=sys.stderr)
    raise
print("TensorRT plan saved :", plan_p)
