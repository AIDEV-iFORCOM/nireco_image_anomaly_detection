#!/usr/bin/env python3
# ============================================================
# retrain.py - Jetson Orin Nano Retraining Batch
#              (PatchCore / PaDiM / EfficientAD / ConvVAE)
# ============================================================
"""
python retrain.py \
  --model patchcore \
  --pretrained_pth models/PatchCore/model.pth \
  --meta_json      models/PatchCore/meta.json \
  --category       VisA_pipe_fryum \
  --data_root      data \
  --epochs         3 \
  --batch          4 \
  --out_dir        models/PatchCore/retrained \
  --load_bank      models/PatchCore/memory_bank.npz  # PatchCore only
  --log            on|off    # default on
"""
from __future__ import annotations
import argparse, json, sys, logging, os, torch.multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Literal

import torch
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from pytorch_lightning import Trainer
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim, EfficientAd
from anomalib.data import Folder
from model_class.vae import ConvVAE

# ---------- CUDA / mem stabilization -------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = False
mp.set_start_method("spawn", force=True)

# ---------- meta.json schema ---------------------------------
class PatchCoreMeta(BaseModel):
    backbone: str; layers: List[str]; coreset_ratio: float = Field(gt=0, le=1)
    image_size: int; weights_pth: str; memory_bank: str; 
    image_threshold: float; image_threshold_auto: float; 
    #raw_image_min: float; raw_image_max: float;

class PaDiMMeta(BaseModel):
    backbone: str; layers: List[str]; n_features: int; 
    image_size: int; weights_pth: str; stats: str; 
    image_threshold: float; image_threshold_auto: float; 
    #raw_image_min: float; raw_image_max: float;

class EfficientADMeta(BaseModel):
    model_size: Literal["small", "medium"]; learning_rate: float
    image_size: int; weights_pth: str; 
    image_threshold: float; image_threshold_auto: float; 
    #raw_image_min: float; raw_image_max: float;

class VAEMeta(BaseModel):
    latent_dim: int; learning_rate: float; 
    image_size: int; weights_pth: str; 
    image_threshold: float; image_threshold_auto: float; 
    #raw_image_min: float; raw_image_max: float; 

SCHEMA: Dict[str, type[BaseModel]] = {
    "patchcore": PatchCoreMeta, "padim": PaDiMMeta,
    "efficientad": EfficientADMeta, "vae": VAEMeta,
}

def load_meta(path: Path, name: str) -> BaseModel:
    try:
        return SCHEMA[name](**json.loads(path.read_text()))
    except ValidationError as e:
        sys.stderr.write(f"[meta.json ERROR]\n{e}\n"); sys.exit(1)

# ---------- CLI ----------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(SCHEMA))
ap.add_argument("--pretrained_pth", required=True, type=Path)
ap.add_argument("--meta_json",     required=True, type=Path)
ap.add_argument("--category",      required=True)
ap.add_argument("--data_root",     required=True, type=Path)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch",  type=int, default=4)
ap.add_argument("--out_dir", type=Path, default=Path("runs/retrain"))
ap.add_argument("--load_bank", type=Path, default=None)
ap.add_argument("--load_stats", type=Path, default=None) 
ap.add_argument("--log", choices=["on", "off"], default="on")
args = ap.parse_args()

# ---------- Logger -------------------------------------------
if args.log == "on":
    log_path = (Path(__file__).resolve().parent / "logs" /
                f"train_{args.model}.log")
    log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ], force=True,
    )
else:
    logging.basicConfig(handlers=[logging.NullHandler()], force=True)

logger = logging.getLogger("retrain")
logger.info("===== retrain.py start =====")

# ---------- utils --------------------------------------------
MODEL_REGISTRY = {
    "patchcore": Patchcore, "padim": Padim,
    "efficientad": EfficientAd, "vae": ConvVAE,
}

def build_folder_dm(cat: str, root: Path, batch: int) -> Folder:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
            ".ppm", ".pgm", ".webp", ".JPG", ".JPEG", ".PNG",
            ".BMP", ".TIF", ".TIFF", ".PPM", ".PGM", ".WEBP")
    return Folder(
        name=cat, root=root, normal_dir=f"train/{cat}",
        abnormal_dir=f"test/{cat}/anomaly",
        normal_test_dir=f"test/{cat}/normal",
        train_batch_size=batch, eval_batch_size=batch,
        num_workers=0, extensions=exts,
    )

def filter_kwargs(name: str, meta: BaseModel) -> dict:
    allowed = {
        "patchcore": {"backbone", "layers", "coreset_ratio"},
        "padim":     {"backbone", "layers", "n_features"},
        "efficientad": {"model_size"},
    }
    if name == "vae": return {}
    kw = meta.dict()
    if name == "patchcore":
        kw["coreset_sampling_ratio"] = kw.pop("coreset_ratio")
    if name == "efficientad":
        kw["pretrained"] = False
    return {k: kw[k] for k in allowed[name] if k in kw}

# ---------- execution ----------------------------------------
meta = load_meta(args.meta_json, args.model)
args.out_dir.mkdir(parents=True, exist_ok=True)

if args.model == "vae":
    model = ConvVAE(meta.latent_dim, meta.learning_rate, meta.image_size)

    dm = build_folder_dm(args.category, args.data_root, args.batch)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=False
    ).fit(model,
          train_dataloaders=train_loader,
          val_dataloaders=val_loader)

else:
    core = MODEL_REGISTRY[args.model](**filter_kwargs(args.model, meta))
    dm = build_folder_dm(args.category, args.data_root, args.batch)
    core.load_state_dict(torch.load(args.pretrained_pth, map_location="cpu"), strict=False)
    logger.info("Loaded weights: %s", args.pretrained_pth)

    # EfficientAD: skip imagenette teacher download
    if args.model == "efficientad":
        if hasattr(core, "prepare_imagenette_data"):
            core.prepare_imagenette_data = lambda *a, **k: None
        core.imagenet_dir = ""

        # -------- disavle teache ------------------------
        core.teacher_model = None
        core.teacher_loss  = None

        dmy = torch.zeros(1, 3, meta.image_size, meta.image_size)
        core.imagenet_loader   = [(dmy,)]
        core.imagenet_iterator = iter(core.imagenet_loader)

    # ----- PatchCore memory_bank load ------------------------
    if args.model == "patchcore" and args.load_bank and args.load_bank.exists():
        target = core.model if hasattr(core, "model") else core
        for fn in ("load_memory_bank", "load_memory"):
            if hasattr(target, fn):
                getattr(target, fn)(args.load_bank)
                logger.info("Loaded memory_bank: %s", args.load_bank); break
            
    # ----- PaDiM stats (mu, inv_cov) load -------------------------
    if args.model == "padim" and args.load_stats and args.load_stats.exists():
        stats = np.load(args.load_stats)
        g = getattr(core.model, "gaussian", None)
        if g is not None:
            g.register_buffer("mean", torch.tensor(stats["mu"]))
            g.register_buffer("inv_covariance", torch.tensor(stats["cov_inv"]))
            g.is_fitted = True
            logger.info("Loaded stats: %s", args.load_stats)
        else:
            logger.warning("PaDiM gaussian not found - stats not loaded")

    Engine(max_epochs=args.epochs, accelerator="gpu").fit(core, dm)
    model = core

# ---------- save ----------
torch.save(model.state_dict(), args.out_dir / "model.pth")
logger.info("Saved model.pth to %s", args.out_dir)

# ---------- PatchCore: memory_bunk ----------
if args.model == "patchcore":
    target = model.model if hasattr(model, "model") else model
    mb_path = args.out_dir / "memory_bank.npz"

    for fn in ("save_memory_bank", "save_memory"):
        if hasattr(target, fn):
            getattr(target, fn)(mb_path)
            logger.info("Saved memory_bank via %s -> %s", fn, mb_path)
            break
    else:
        for attr in ("memory_bank", "_memory_bank"):
            if hasattr(target, attr):
                mb = getattr(target, attr)

                np.savez_compressed(mb_path, embeddings=mb.detach().cpu().numpy()
                                    if torch.is_tensor(mb) else mb)
                logger.info("Saved memory_bank (fallback) to %s", mb_path)
                break
        else:
            logger.warning("No memory_bank found - nothing saved")

# ---------- PaDiM: stats(mu, cov_inv) ----------
if args.model == "padim":
    g = getattr(model.model, "gaussian", None)
    if g is not None and hasattr(g, "inv_covariance"):
        mu      = g.mean.detach().cpu().numpy()            # shape: (n_features,)
        cov_inv = g.inv_covariance.detach().cpu().numpy()  # shape: (n_features, n_features)

        stats_path = args.out_dir / "stats.npz"
        np.savez_compressed(stats_path, mu=mu, cov_inv=cov_inv)
        logger.info("Saved stats.npz to %s", stats_path)
    else:
        logger.warning("PaDiM gaussian.inv_covariance not found – stats.npz not saved")

# ---------- cleanup ----------
del dm, model
torch.cuda.empty_cache()
logger.info("Resources freed - retrain.py completed.")
