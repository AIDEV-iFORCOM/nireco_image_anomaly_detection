from __future__ import annotations

# 標準ライブラリ
from collections import defaultdict
from pathlib import Path
from typing import Sequence

# サードパーティ
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torchvision.transforms.functional import to_tensor
from pytorch_lightning import LightningModule

# -------- VAEモデル定義 --------
class ConvVAE(LightningModule):
    """
    * 入力: 3×IMAGE_SIZE×IMAGE_SIZE (0–1 float)
    * 潜在: latent_dim (Optuna で探索)
    """
    def __init__(self, latent_dim: int = 128, lr: float = 1e-3, img_size: int = 256):
        super().__init__()
        self.save_hyperparameters()

        chs = [3, 32, 64, 128]
        enc = []
        for cin, cout in zip(chs, chs[1:]):
            enc += [nn.Conv2d(cin, cout, 4, 2, 1), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*enc)

        # --- 入出力次元を動的に算出 ------------------------------------
        with torch.no_grad():
            dummy   = torch.zeros(1, 3, img_size, img_size)
            enc_out = self.encoder(dummy)
        c, h, w     = enc_out.shape[1:]
        flat_dim    = c * h * w

        self._enc_shape = (c, h, w)           # decode 時に使用

        self.mu     = nn.Linear(flat_dim, latent_dim)
        self.logvar = nn.Linear(flat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        dec = [
            nn.ConvTranspose2d(c, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  3, 4, 2, 1), nn.Sigmoid()
        ]
        self.decoder = nn.Sequential(*dec)
        self._img_size = img_size

    # ----- forward 系 -----
    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + torch.randn_like(std)*std

    def decode(self, z):
        c, h, w = self._enc_shape
        h = self.fc_dec(z).view(z.size(0), c, h, w)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # ---------- 画像Tensor整形 ----------
    def _prepare_x(self, batch):
        # (以下コメントもコードも元のまま: 処理は同じ)
        if isinstance(batch, (list, tuple)):
            img = batch[0]
        elif isinstance(batch, dict):
            img = batch.get("image", next(iter(batch.values())))
        else:
            img = batch

        cls_name = img.__class__.__name__
        if cls_name.endswith("ImageBatch"):
            img = list(img)
        elif cls_name.endswith("ImageItem"):
            if hasattr(img, "tensor"):
                img = img.tensor
            elif hasattr(img, "data"):
                img = img.data
            elif hasattr(img, "image"):
                img = img.image

        if isinstance(img, Sequence) and not isinstance(img, (torch.Tensor, np.ndarray)):
            tensors = [self._prepare_x([sub]) for sub in img]
            return torch.cat(tensors, dim=0)

        if isinstance(img, torch.Tensor):
            x = img
        elif isinstance(img, np.ndarray):
            if img.dtype == object:
                x = torch.stack([to_tensor(el) for el in img], dim=0)
            else:
                x = torch.from_numpy(img)
        else:
            x = to_tensor(img)

        if x.dtype == torch.uint8:
            x = x.float().div_(255)
        elif x.is_floating_point() and x.max() > 1.5:
            x = x.div_(255)

        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()

        if x.shape[-2:] != (self._img_size, self._img_size):
            x = F.interpolate(x, size=(self._img_size, self._img_size),
                              mode="bilinear", align_corners=False)

        return x.contiguous()
        
    def _vae_loss(self, x_hat, x, mu, logvar):
        recon = F.mse_loss(x_hat, x, reduction="mean")
        kl    = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kl*0.0005

    def _shared_step(self, batch, stage):
        x = self._prepare_x(batch)
        x_hat, mu, logvar = self(x)
        loss = self._vae_loss(x_hat, x, mu, logvar)
        self.log(f"{stage}_loss", loss, prog_bar=False)
        return loss, x, x_hat

    def training_step(self, batch, _):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, _):
        # --- 画像を取り出す -------------------------------------------
        if isinstance(batch, dict):
            img_raw   = batch["image"]                 # 画像 Tensor
            path_list = batch.get("image_path", [""])  # list[str|Path] or str
            if isinstance(path_list, (str, Path)):
                path_list = [path_list]

            # 正常=0 / 異常=1 のラベルを作成
            y = torch.tensor(
                [1.0 if ("anomaly" in str(p) or "defect" in str(p)) else 0.0
                 for p in path_list],
                device=self.device,
                dtype=torch.float32,
            )

        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            img_raw, y = batch                         # (img, label)
            y = y.float().to(self.device)

        else:                                          # ラベル情報なし → 正常扱い
            img_raw = batch
            # len(img_raw) でバッチ長を取得（ImageBatch に対応）
            y = torch.zeros(len(img_raw), device=self.device)

        # (_shared_step は画像だけ渡せば OK)
        loss, x, x_hat = self._shared_step(img_raw, "val")

        errs = F.mse_loss(x_hat, x, reduction="none").mean([1, 2, 3])

        # --- バッファに蓄積 -------------------------------------------
        if not hasattr(self, "_val_buf"):
            self._val_buf = defaultdict(list)
        self._val_buf["errs"].append(errs.detach())
        self._val_buf["labels"].append(y.detach())

    def on_validation_epoch_end(self):
        if not hasattr(self, "_val_buf") or len(self._val_buf["errs"]) == 0:
            return

        errs   = torch.cat(self._val_buf["errs"])
        labels = torch.cat(self._val_buf["labels"]).view(-1)

        # --- AUROC 計算（正常=0, 異常=1） ----------------------------
        if labels.unique().numel() < 2:
            auroc = torch.tensor(0.5, device=self.device)
        else:
            auroc = torch.tensor(
                roc_auc_score(labels.cpu(), errs.cpu()),
                device=self.device
            )

        self.log("val_AUROC", auroc, prog_bar=True)

        # --- デバッグ用ログ -----------------------------------------
        n_norm = int((labels == 0).sum())
        n_anom = int((labels == 1).sum())
        self.print(                                # ← Lightning の rank_zero_only print
            f"[VAL] epoch={self.current_epoch}  normals={n_norm}  "
            f"anomalies={n_anom}  AUROC={auroc.item():.4f}"
        )

        self._val_buf.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ------- Tensor化 (共通関数) -------
def _to_tensor(img):
    """PIL / np.ndarray / torch.Tensor いずれも Tensor 化 (0-1)"""
    if isinstance(img, torch.Tensor):
        return img.float() / (255 if img.dtype == torch.uint8 else 1)
    import numpy as np
    from torchvision.transforms.functional import to_tensor
    if isinstance(img, np.ndarray):
        return torch.from_numpy(img).float() / (255 if img.dtype != np.float32 else 1)
    return to_tensor(img)           # PIL など

# ------- 画像&ラベル取出 (共通関数) -------
def extract_xy(
    batch,
    device: torch.device | str = "cpu",
    *,
    return_counts: bool = False,
    return_paths:  bool = False,
):
    """
    ImageBatch / list[ImageItem] / dict のいずれが来ても以下を返す共通関数
        x : Tensor[B,3,H,W]  (float32 0–1, device 移動済み)
        y : Tensor[B]        (0:normal, 1:anomaly, float32, device 移動済み)
        [counts]             n_norm, n_anom   ※ return_counts=True の時
        [paths]              list[str|Path]   ※ return_paths =True の時

    Parameters
    ----------
    batch : dict | ImageBatch | Sequence
        DataLoader から受け取るバッチ
    device : torch.device | str
        `.to(device)` 先
    return_counts : bool, default False
        True のとき (n_norm, n_anom) を返す
    return_paths : bool, default False
        True のとき 画像パスの list を返す
    """
    # ------ anomalib >=0.8 形式の dict -------
    if isinstance(batch, dict):
        x = batch["image"]
        y = batch["label"].float()
        paths = batch.get("image_path", [""] * len(y))

    # ------ ImageBatch (list-like) または list[ImageItem] -------
    else:
        items: Sequence = list(batch)
        x = torch.stack([_to_tensor(getattr(it, "tensor", it)) for it in items])
        # ― ラベル取得 ―
        if hasattr(items[0], "label"):
            y = torch.tensor([float(it.label) for it in items])
        else:
            paths = [str(getattr(it, "image_path", "")) for it in items]
            y = torch.tensor([1.0 if ("anomaly" in p or "defect" in p) else 0.0 for p in paths])

    # ------ 後処理／戻り値組み立て -------
    x, y = x.to(device), y.to(device)
    out  = [x, y]

    if return_counts:
        n_norm = int((y == 0).sum())
        n_anom = int((y == 1).sum())
        out.extend([n_norm, n_anom])

    if return_paths:
        out.append(paths)

    return tuple(out) if len(out) > 1 else out[0]
