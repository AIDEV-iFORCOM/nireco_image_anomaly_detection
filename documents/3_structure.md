## `ディレクトリ構造`
### 全体構造
<br>

| ディレクトリ名                       | 説明                                              | 実行対象      |
|-------------------------------------|--------------------------------------------------|----------------------|
| **`1_train_jupyterlab_docker/`**    | 初期学習用ディレクトリ (Docker)                        | PC                   |
| **`2_jetson_docker/`**              | 再学習/TensorRT化用ディレクトリ (Docker)       | Jetson               |
| **`3_jetson_infer/`**               | 推論実行プログラム               | Jetson               |
| **`data/`**                         | 学習データ                                       | PC/Jetson共通        |
| **`models/`**                       | モデル格納場所                           | PC/Jetson共通        |
---
### ディレクトリ・ファイル説明
<br>

#### 1. 初期学習ディレクトリ (PC / Docker) {#1_train_jupyterlab_docker}
- **`1_train_jupyterlab_docker/`** 
    - `docker-compose.yml`
    - `docker/`
        - `Dockerfile`
        - `requirements.txt`
    - `work/`
        - `script/`
            - `PatchCore.ipynb`　　　　　　　： 初期学習スクリプト (モデル別)
            - `PaDim.ipynb`
            - `EfficientAD.ipynb`
            - `VAE.ipynb`
---
#### 2. 再学習/TensorRT化ディレクトリ (Jetson / Docker) {#2_jetson_docker}

- **`2_jetson_docker/`**
    - `docker-compose.yml`
    - `docker/`
        - `Dockerfile`
        - `requirements.txt`
    - `work/`
        - `retrain.py`　　　　　　　　　　　： 再学習プログラム
        - `convtrt.py`　　　　　　　　　　　： ONNXエクスポート & TensorRTエンジン化プログラム
        - `model_class/`
            - `vae.py`　　　　　　　　　　　： VAEモデルクラスファイル (Anomalibに存在しないVAEのみ必要)
---
#### 3. 推論実行ディレクトリ (Jetson)  {#3_jetson_infer}

- **`3_jetson_infer/`**
    - `patchcore`　　　　　　　　　　　　　　： TensorRT実行バイナリ (モデル別)
    - `padim`
    - `efficientad`
    - `vae`<br><br>
    - +`result_<model_name>.csv`　　　　　　　： 実行結果テキストファイル (推論実行後に作成)
---
#### 画像データディレクトリ (PC / Jetson共通) {#data}

- **`data/`**
    - `shellscript/`
        - `download_VisA.sh`　　　　　　　　　： VisAデータセットのダウンロードシェルスクリプト
        - `move_VisA.sh`　　　　　　　　　　　 ： VisAデータセットの学習・検証用配置シェルスクリプト
    - `download/`
        - `VisA`　　　　　　　　　　　　　　　 ： VisAデータセットのダウンロード先
            - `<image_category>`
    - `test/`　　　　　　　　　　　　　　　　　： 検証用データ　※正常・異常あり
        - `<dataset_name>_<image_category>/`　　 ： 例）VisA_pipe_fryum → ここが、学習時の設定 `CATEGORY` と対応します
            - `normal/`
            - `anomaly/`
    - `train/`　　　　　　　　　　　　　　　　 ： 学習用データ　※正常のみ
        - `<dataset_name>_<image_category>/`
---
#### モデル格納ディレクトリ (PC / Jetson共通) {#models}

- **`models/`**
    - `PatchCore/`
    - `PaDiM/`
    - `EfficientAD/`
    - `VAE/`<br><br>

- ※ `models/`ディレクトリは、PCとJetsonではモデル名`<model_name>/`ディレクトリ配下の構成が異なります。

##### PC (JupyterLab) の場合 {#models-pc}
- **`<model_name>/`**
    - `<image_category>/`　　　　　　　　　　　　　： 画像のカテゴリ名
        - `<YYYYMMDD_HHMMSS>/`　　　　　　　　　　　： 学習実行時のシステム日時
            - `checkpoint/`　　　　　　　　　　　　　
                - `best.ckpt`　　　　　　　　　　　： チェックポイントファイル
            - `logs/`　　　　　　　　　　　　　　　： 学習ログファイル格納
            - `param/`
                - `best_params.yaml`　　　　　　　　 ： 最適ハイパーパラメーター
                - `search_space.yaml`　　　　　　　　： 探索範囲 (サーチスペース)
                - `trials.csv`　　　　　　　　　　　 ： 探索履歴
            - **`pytorch/`**　： **学習済ファイル格納場所 >>>>> このフォルダの中身をJetsonへ移動します**
                - **`meta.json`**　　　　　　　　　　　　☆ 学習メタデータ
                - **`model.pth`**　　　　　　　　　　　　☆ 最終重みファイル
                - **`stats.npz`**　　　　　　　　　　　　☆ 平均 μ & 逆共分散 Σ ※PaDiMのみ
                - **`memory_bank.npz`**　　　　　　　　 　☆ メモリバンク ※PatchCoreのみ
            - `test_result/`  　　　　　　　　　　　　： 性能検証結果データ(`data/test/`と対応します)<br>
                - `predictions.csv`　　　　　　　　　： 画像毎の異常スコアと閾値を用いた判断結果
                    - `images/`
                        - `normal/`　　　　　　　　　 ： 正常画像のアノマリーマップと異常検知マスク(参考)
                        - `anomaly/`　　　　　　　　　： 異常画像のアノマリーマップと異常検知マスク(参考)

※チェックポイントファイル(.ckpt)を利用して再学習を行うことも可能ですが、今回は .pth ファイル + メタデータ の組合せを利用しております。

##### Jetson の場合  {#models-jetson}
- **`model_name/`**
    - `meta.json`　　　　　　　　　　　　　　　　　　★ 学習メタデータ (PCから移動)
    - `model.pth`　　　　　　　　　　　　　　　　　　★ 最終重みファイル (PCから移動)
    - `stats.npz`　　　　　　　　　　　　　　　　　　★ 平均 μ & 逆共分散 Σ (PCから移動)
    - `memory_bank.npz`　 　　　　　　　　　　　　　　★ メモリバンク(PCから移動)<br><br>
    - `engine/(例)`
        - **`model.plan`**　　　　　　　　　　　　　　　： TensorRTエンジンファイル **※推論に使用されます**
        - `model.onnx`　　　　　　　　　　　　　　　： ONNXエクスポートファイル<br><br>
    - `retrain(例)`　　　　　　　　　　　　 　　　　 ： Jetson上での再学習後ファイル格納(meta.jsonは更新なし) 
        - `model.pth`
        - `stats.npz`
        - `memory_bank.npz`　　　　　　　　　　　　 ： メモリバンク **※推論に使用されます**
        <br>

※入力ファイルや変換ファイルの出力先は、コマンドパラメータによって任意のパスに変更可能です。
　([再学習コマンド](2_models.md#comretrain)、[TensorRT化コマンド](2_models#comtensorrt)、[推論コマンド](2_models#cominfer) 参照)