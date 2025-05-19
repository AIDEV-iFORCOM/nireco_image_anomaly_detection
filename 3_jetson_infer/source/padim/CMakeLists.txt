cmake_minimum_required(VERSION 3.18)
project(padim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# パッケージ
find_package(OpenCV REQUIRED)          # ex: 4.8.0
find_package(CUDAToolkit REQUIRED)     # 12.6.68 (JetPack 6 DP)
find_package(ZLIB REQUIRED)

# TensorRT (JetPack 6.2)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include /usr/include/aarch64-linux-gnu)
find_library(TENSORRT_LIB_NVINFER nvinfer PATHS /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(TENSORRT_LIB_PARSER nvonnxparser PATHS /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(TENSORRT_LIB_PLUGIN nvinfer_plugin PATHS /usr/lib /usr/lib/aarch64-linux-gnu)

if(NOT (TENSORRT_INCLUDE_DIR AND TENSORRT_LIB_NVINFER))
    message(FATAL_ERROR "TensorRT headers or libs not found")
endif()

add_library(TensorRT::nvinfer    SHARED IMPORTED)
add_library(TensorRT::onnxparser SHARED IMPORTED)
add_library(TensorRT::plugin     SHARED IMPORTED)

set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION             "${TENSORRT_LIB_NVINFER}"
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}")
set_target_properties(TensorRT::onnxparser PROPERTIES
    IMPORTED_LOCATION             "${TENSORRT_LIB_PARSER}"
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}")
set_target_properties(TensorRT::plugin PROPERTIES
    IMPORTED_LOCATION             "${TENSORRT_LIB_PLUGIN}"
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}")

# FAISS (libfaiss.so GPU 同梱)
find_library(FAISS_LIB NAMES faiss PATHS /usr/local/lib REQUIRED)

# cnpy (ヘッダ + cpp)
add_library(cnpy STATIC cnpy/cnpy.cpp)

# 実行ファイル
add_executable(padim main.cpp)

# インクルードパス
target_include_directories(padim PRIVATE
    cnpy
    ${CMAKE_SOURCE_DIR}/third_party      # cxxopts.hpp
    ${CUDAToolkit_INCLUDE_DIRS}
    /usr/local/include                   # faiss headers
)

# リンク
target_link_libraries(padim
    cnpy
    ${FAISS_LIB}                             # /usr/local/lib/libfaiss.so
    TensorRT::nvinfer
    TensorRT::onnxparser
    TensorRT::plugin
    CUDA::cudart CUDA::cublas                # cuBLAS
    ZLIB::ZLIB
    ${OpenCV_LIBS}
)

# 3. オプション（任意）
# Orin sm_87 固定。サイズ削減したい場合のみ設定
# set(CMAKE_CUDA_ARCHITECTURES 87)

# Release ビルド最適化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
