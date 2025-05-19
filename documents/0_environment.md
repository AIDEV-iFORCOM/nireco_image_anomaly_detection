## `環境構築`
## PC 環境構築

#### 前提
<br>

##### 利用する docker イメージ は、以下の環境での互換性を確認済です
`cuda:12.2.2-cudnn8-runtime-ubuntu22.04`

##### OS

```
ubuntu22.04 (WSL2)
```

##### NVIDIA Driver / CUDA
```
 NVIDIA Driver Version: 53*  CUDA 12.2
 NVIDIA Driver Version: 55*  CUDA 12.4
 NVIDIA Driver Version: 56*  CUDA 12.6
 NVIDIA Driver Version: 57*  CUDA 12.8
```

---
## Jetson 環境構築

#### 前提
`JetPack 6.2 / TensorRT 10.3 / CUDA12.6`

#### 事前のインストールモジュール
```
apt-get install -y lrzsz　　　　　　　　　　　　　 # PCからのファイル転送
apt-get install -y nano vim　　　　　　　　　　　  # エディタ
apt-get install -y build-essential cmake         # 標準cmake
```

##### 以下の手順でGPU動作するTensorRT、C++バイナリのビルド可能環境を構築します
- (1) Kitware APT で cmake をアップグレード
- (2) FAISS v1.10.0 インストール

---

### (1) Kitware APT で cmake をアップグレード
<br>

##### 署名鍵
```
sudo apt-get update
sudo apt-get install -y ca-certificates gpg wget
```
##### 鍵を取得
```
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | gpg --dearmor - \
  | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
```

##### リポジトリ追加（JetPack 6.2 = Ubuntu 22.04 jammy）
```
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" \
 | sudo tee /etc/apt/sources.list.d/kitware.list
```

##### 更新して cmake を入れ替える
```
sudo apt-get update
sudo apt-get install -y kitware-archive-keyring cmake
```

##### 確認コマンド
```
cmake --version
```
##### 結果
```
cmake version 4.0.1
CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

---
### (2) FAISS v1.10.0 インストール
<br>

#### まず nvcc のパスを通す
##### 共通シンボリックリンク
```
sudo ln -sf /usr/local/cuda-12.6 /usr/local/cuda
```
##### PATH と LD_LIBRARY_PATH を恒久設定
```echo 'export PATH=/usr/local/cuda/bin:$PATH'                 >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
##### 確認コマンド
```
nvcc --version
```
##### 結果
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:14:07_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
```

#### FAISS v1.10.0 インストール
```
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake ninja-build \
    libopenblas-dev         \
    libgflags-dev           \
    git python3-dev swig
git clone --branch v1.10.0 --depth 1 https://github.com/facebookresearch/faiss.git
cd faiss
```

##### 共有ライブラリを ON にして Cmake
```
cmake -S . -B build -G Ninja \
      -DFAISS_ENABLE_GPU=ON \
      -DFAISS_ENABLE_C_API=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DCMAKE_BUILD_TYPE=Release
```

##### ビルド & インストール
```
cmake --build build -j$(nproc)
sudo cmake --install build
sudo ldconfig
```

##### 確認コマンド
```
ldconfig -p | grep libfaiss
```
##### 結果
```
-> libfaiss.so (libc6,AArch64) => /usr/local/lib/libfaiss.so
```

---
#### （参考）C++ プログラムのビルド方法
<br>

##### C++プログラムの存在するディレクトリへ cnpy と cxxopts ヘッダーを取得   
```
cd project 
```
##### cnpy をサブフォルダへクローン
```
git clone https://github.com/rogersce/cnpy.git cnpy
```

##### cxxopts ヘッダーを保存
```
mkdir -p third_party
wget -qO cxxopts.hpp https://raw.githubusercontent.com/jarro2783/cxxopts/master/include/cxxopts.hpp
```

##### CMakeLists.txt
```
参考ファイルを同梱
```

##### ビルドコマンド
```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```