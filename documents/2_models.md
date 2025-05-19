## `モデル`
### <a id="modellist">異常検知モデル一覧</a>

| モデル名                               | ライブラリ   | ハイパーパラメータ                        | バックボーン                      | 学習対象                               | 推論時に必要                             |
| -------------------------------------- | ------------ | ----------------------------------------- | --------------------------------- | -------------------------------------- | ---------------------------------------- |
| PatchCore                              | Anomalib     | backbones,<br>layers,<br>coreset         | resnet18,<br>resnet34,<br> etc.        | 正常パッチ埋込<br>メモリバンク          | model.plan <br>+ memory_bank.npz            |
| PaDiM                                  | Anomalib     | backbones,<br>layers<br><br>                     | resnet18,<br>wide_resnet50_2,<br>etc. | 多変量ガウス分布 (μ, Σ)                | model.plan のみ<br>（.planへ埋込）  |
| EfficientAD                            | Anomalib     | model_size,<br>learning_rate<br><br>             | EfficientNetのみ（固定）          | 教師ネットワーク<br>AutoEncoder重み     | model.plan のみ                          |
| VAE <br>(Variational Autoencoder)          | なし         | latent_dimension,<br>learning_rate<br><br>       | なし (本体がAutoEncoder)          | モデル重み                             | model.plan のみ                          |


| モデル名                               | Jetson 上の再学習に必要                            | ターゲット                                   | 長所                             | 短所                                    |
| -------------------------------------- | ------------------------------------------------- | -------------------------------------------- | -------------------------------- | --------------------------------------- |
| PatchCore                              | model.pth<br>+ memory_bank.npz<br>+ meta.json     | 1モデル1カテゴリ                             | 局所異常に強い<br>最高精度         | GPU/RAM要求が大きい                    |
| PaDiM                                  | model.pth<br>+ stats.npz<br>+ meta.json           | 1モデル1カテゴリ                             | 局所異常に強い<br>最高精度         | GPU/RAM要求が大きい                    |
| EfficientAD                            | model.pth<br>+ meta.json<br><br>                  | 1モデル1カテゴリ<br>(上記より横断し易い)   | 高速<br>論理異常にも対応         | バランス型                             |
| VAE <br>(Variational Autoencoder)      | model.pth<br>+ meta.json<br><br>                  | 1モデル1カテゴリ<br>(改良版は汎化性能あり) | 最高速、汎化性能あり<br>多数の派生モデル | 局所異常の検知に弱い             |
---

### <a id="meta">学習メタファイル (meta.json)</a>

##### PatchCore 例
```
{
  "backbone": "resnet34",　　　　　　　　　　　　： ハイパーパラメータ1(バックボーンCNNモデル)
  "layers": [　　　　　　　　　　　　　　　　　　 ： ハイパーパラメータ2(使用レイヤー)
    "layer2"
  ],
  "coreset_ratio": 0.05,　　　　　　　　　　　　　： ハイパーパラメータ3(coreset_ratio)
  "image_size": 256,　　　　　　　　　　　　　　　： 画像サイズ
  "weights_pth": "model.pth",　　　　　　　　　　： 学習済モデル重みファイル名
  "memory_bank": "memory_bank.npz",　　　　　　　： メモリバンクファイル名 (PatchCoreのみ)
  "image_threshold": 0.5,　　　　　　　　　　　　 ： 手動で設定した閾値
  "image_threshold_auto": 8.257281303405762,　　： 自動で設定された閾値(検証処理必要)
  "raw_image_min": 6.724304676055908,　　　　　　： 正規化前の最小の異常スコア
  "raw_image_max": 20.64291000366211　　　　　　 ： 正規化前の最大の異常スコア
}
```
##### PaDiM 例
```
{
  "backbone": "resnet18",　　　　　　　　　　　　　： ハイパーパラメータ1(バックボーンCNNモデル)
  "layers": [　　　　　　　　　　　　　　　　　　　 ： ハイパーパラメータ2(使用レイヤー)
    "layer2"
  ],
  "n_features": 128,　　　　　　　　　　　　　　　　： ハイパーパラメータ3(n_features)
  "image_size": 256,　　　　　　　　　　　　　　　　： 画像サイズ
  "weights_pth": "model.pth",　　　　　　　　　　　： 学習済モデル重みファイル名
  "stats": "stats.npz",　　　　　　　　　　　　　　 ： 多変量ガウス分布ファイル名 (PaDiMのみ)
  "image_threshold": 0.5,　　　　　　　　　　 　 　 ： 手動で設定した閾値
  "image_threshold_auto": 28.416128158569336,　 　： 自動で設定された閾値(検証処理必要)
  "image_min": 12.350069999694824,　　　　 　　　　： 正規化前の最小の異常スコア
  "image_max": 117.86776733398438　　　　　　　　　： 正規化前の最大の異常スコア
}
```
##### EfficientAD 例
```
{
  "model_size": "medium",　　　　　　　　　　　　　： ハイパーパラメータ1(モデルサイズ)
  "learning_rate": 0.0001,　　　　　　　　　　　 　： ハイパーパラメータ2(学習率)
  "image_size": 256,　　　　　　　　　　　　　　　　： 画像サイズ
  "weights_pth": "model.pth",　　　　　　　　　　　： 学習済モデル重みファイル名
  "image_threshold": 0.5,　　　　　　　　　　 　　 ： 手動で設定した閾値
  "image_threshold_auto": 0.10265577584505081,　 ： 自動で設定された閾値(検証処理必要)
  "raw_image_min": 0.03536586835980415,　　　　   ： 正規化前の最小の異常スコア
  "raw_image_max": 0.6505160927772522　　　　　 　： 正規化前の最大の異常スコア
}
```
##### VAE 例
```
{
  "latent_dim": 128,　　　　　　　　　　　　　　　　： ハイパーパラメータ1(latent_dim)
  "learning_rate": 0.001,　　　　　　　　　　　　 　： ハイパーパラメータ2(学習率)
  "image_size": 256,　　　　　　　　　　　　　　 　　： 画像サイズ
  "weights_pth": "model.pth",　　　　　　　　　 　　： 学習済モデル重みファイル名
  "image_threshold": 0.5,　　　　　　　　　　 　 　 ： 手動で設定した閾値
  "image_threshold_auto": 0.009446951560676098,　 ： 自動で設定された閾値(検証処理必要)
  "raw_image_min": 0.00648981099948287,　　　　    ： 正規化前の最小の異常スコア
  "raw_image_max": 0.015051797963678837　　　　  　： 正規化前の最大の異常スコア
}
```

---

### <a id="comretrain">再学習コマンド</a>
<br>

##### 説明
`python3` **`retrain.py`**
```
  --model 　　　　　　　　　 ： 対象モデル (patchcore/padim/efficientad/vae)
  --pretrained_pth  　　　　： 事前学習重みファイル(.pth)
  --meta_json    　　　　　 ： 学習メタファイル
  --category  　　　　　 　 ： 学習画像カテゴリ
  --data_root 　　　　　　　 ： 画像データルートフォルダ
  --epochs　　　　　　　　　 ： 学習エポック数
  --batch          　　　　 ： バッチサイズ
  --out_dir　　　　　　　　　： 出力先ディレクトリ
  --load_bank  　　　　　　　： メモリバンク (PatchCoreのみ)
  --load_stats　　　　　　　 ： 多変量ガウス分布 (PaDiMのみ)
  --log            　　　　　： ログ出力 on/off (default on)
```

##### PatchCore 例

```
python3 retrain.py \
  --model patchcore \
  --pretrained_pth models/PatchCore/model.pth \
  --meta_json models/PatchCore/meta.json \
  --category VisA_pipe_fryum \
  --data_root data \
  --epochs 1 \
  --batch 4 \
  --out_dir models/PatchCore/retrained \
  --load_bank models/PatchCore/memory_bank.npz
```

##### PaDiM 例

```
python3 retrain.py \
  --model padim \
  --pretrained_pth models/PaDiM/model.pth \
  --meta_json      models/PaDiM/meta.json \
  --category       VisA_pipe_fryum \
  --data_root      data \
  --epochs         1 \
  --batch          4 \
  --out_dir        models/PaDiM/retrained \
  --load_stats     models/PaDiM/stats.npz
```

##### EfficientAD 例

```
python3 retrain.py \
  --model efficientad \
  --pretrained_pth models/EfficientAD/model.pth \
  --meta_json      models/EfficientAD/meta.json \
  --category       VisA_pipe_fryum \
  --data_root      data \
  --epochs         1 \
  --batch          1 \
  --out_dir        models/EfficientAD/retrained
```

##### VAE 例

```
python3 retrain.py \
  --model vae \
  --pretrained_pth models/VAE/model.pth \
  --meta_json      models/VAE/meta.json \
  --category       VisA_pipe_fryum \
  --data_root      data \
  --epochs         1 \
  --batch          4 \
  --out_dir        models/VAE/retrained
```

---

### <a id="comtensorrt">TensorRT化コマンド</a>
<br>

##### 説明
`python3` **`convtrt.py`**
```
--model 　　　　　　　　　 ： 対象モデル (patchcore/padim/efficientad/vae)
--weights_pth  　　　　　　： 最終重みファイル(.pth)
--meta_json    　　　　　　： 学習メタファイル
--memory_bank  　　　　　　： メモリバンク (PatchCoreのみ)
--opset 17  　　　　　　 　： opset バージョン 17 or 18 (JetPack6.2)
--fp16 　　　　　　　　　 　： FP16化する
--workspace　　　　　　　　： default 4096
--out_dir　　　　　　　　　： 出力先ディレクトリ
--use_dla　　　　　　　　　： default OFF (Jetson Oron nano は、DLAなし) 
```
##### PatchCore 例
```
python3 convtrt.py \
  --model patchcore \
  --weights_pth  models/PatchCore/retrained/model.pth \
  --meta_json    models/PatchCore/meta.json \
  --memory_bank  models/PatchCore/retrained/memory_bank.npz \
  --opset 17  \
  --fp16 \
  --workspace 4096 \
  --out_dir    models/PatchCore/engine
```
##### PaDiM 例
```
python3 convtrt.py \
  --model padim \
  --weights_pth  models/PaDiM/retrained/model.pth \
  --meta_json    models/PaDiM/meta.json \
  --stats  models/PaDiM/retrained/stats.npz \
  --opset 17  \
  --fp16 \
  --workspace 4096 \
  --out_dir    models/PaDiM/engine
```
##### EfficientAD 例
```
python3 convtrt.py \
  --model efficientad  \
  --weights_pth  models/EfficientAD/retrained/model.pth \
  --meta_json    models/EfficientAD/meta.json \
  --opset 17  \
  --fp16 \
  --workspace 4096 \
  --out_dir    models/EfficientAD/engine
```
##### VAE 例
```
python3 convtrt.py \
  --model vae  \
  --weights_pth  models/VAE/retrained/model.pth \
  --meta_json    models/VAE/meta.json \
  --opset 17  \
  --fp16 \
  --workspace 4096 \
  --out_dir    models/VAE/engine
```
---
### <a id="cominfer">推論コマンド</a>
<br>

##### 説明

**`./<model_name>`**`(binary_name)`
```
--engine or -e　　　　　： TensorRTエンジンパスを指定(.plan)
--memory_bank or -m　　： メモリバンクのパスを指定(PatchCore のみ)
--data or -d　　　　　　： 画像ファイルパスを指定
--image_size or -i　　 ： 画像リサイズを指定
--threshold or -t　　　： 異常検知スコア閾値
```
<br>

##### PatchCore 例
```
./patchcore \
  -e ../models/PatchCore/engine/model.plan \
  -m ../models/PatchCore/engine/memory_bank.npz \
  -d ../data/test/VisA_pipe_fryum/anomaly \
  -i 256 \
  -t 100
```
##### PaDiM 例
```
./padim \
  -e ../models/PaDiM/engine/model.plan \
  -d ../data/test/VisA_pipe_fryum/anomaly \
  -i 256 \
  -t 100
```
##### EfficientAD 例
```
./efficientad \
  -e ../models/EfficientAD/engine/model.plan \
  -d ../data/test/VisA_pipe_fryum/normal \
  -i 256 \
  -t 100
```
##### VAE 例
```
./vae \
  -e ../models/VAE/engine/model.plan \
  -d ../data/test/VisA_pipe_fryum/anomaly \
  -i 256 \
  -t 100
```