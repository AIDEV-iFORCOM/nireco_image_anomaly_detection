// main.cpp - VAE inference (Jetson Orin) with TensorRT 10.3

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "cxxopts.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>
#include <numeric>
#include <vector>
#include <string>

namespace fs = std::filesystem;
using Clock  = std::chrono::high_resolution_clock;
using msec_d = std::chrono::duration<double, std::milli>;

// ---------------- Timer helper ----------------
struct Timer {
    Clock::time_point t0;
    void tic()         { t0 = Clock::now(); }
    double toc() const { return std::chrono::duration_cast<msec_d>(Clock::now() - t0).count(); }
};

// ---------------- CUDA error check macro ----------------
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t status = (call);                                       \
        if (status != cudaSuccess) {                                       \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status)      \
                      << std::endl;                                        \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// ---------------- TensorRT logger ----------------
class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << '\n';
    }
};

// ---------------- TensorRT context wrapper ----------------
class TrtCtx {
public:
    explicit TrtCtx(const std::string& plan) {
        log_      = std::make_unique<Logger>();
        runtime_.reset(nvinfer1::createInferRuntime(*log_));
        initLibNvInferPlugins(log_.get(), "");

        std::ifstream fin(plan, std::ios::binary);
        if (!fin) throw std::runtime_error("Cannot open engine: " + plan);
        fin.seekg(0, std::ios::end);
        size_t sz = fin.tellg();
        fin.seekg(0);
        std::vector<char> buf(sz);
        fin.read(buf.data(), sz);

        engine_.reset(runtime_->deserializeCudaEngine(buf.data(), sz));
        if (!engine_) throw std::runtime_error("Engine deserialization failed");
        context_.reset(engine_->createExecutionContext());

        int nbIO = engine_->getNbIOTensors();
        for (int i = 0; i < nbIO; ++i) {
            const char* name = engine_->getIOTensorName(i);
            tensorNames_.emplace_back(name);
        }

        // assume binding 0 = input, 1 = output
        inName_  = tensorNames_[0];
        outName_ = tensorNames_[1];

        auto inShape  = engine_->getTensorShape(inName_.c_str());
        auto outShape = engine_->getTensorShape(outName_.c_str());
        inSize_  = std::accumulate(inShape.d,  inShape.d  + inShape.nbDims, 1LL, std::multiplies<int64_t>());
        outSize_ = std::accumulate(outShape.d, outShape.d + outShape.nbDims, 1LL, std::multiplies<int64_t>());

        CUDA_CHECK(cudaMalloc(&dIn_,  inSize_  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut_, outSize_ * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~TrtCtx() {
        cudaFree(dIn_);
        cudaFree(dOut_);
        cudaStreamDestroy(stream_);
    }

    // hIn[inSize_] -> hOut[outSize_]
    void infer(const float* hIn, float* hOut) {
        CUDA_CHECK(cudaMemcpyAsync(dIn_, hIn, inSize_ * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        for (auto &n : tensorNames_) {
            if (n == inName_)       context_->setTensorAddress(n.c_str(), dIn_);
            else                    context_->setTensorAddress(n.c_str(), dOut_);
        }
        context_->enqueueV3(stream_);
        CUDA_CHECK(cudaMemcpyAsync(hOut, dOut_, outSize_ * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    size_t inputSize()  const { return inSize_; }
    size_t outputSize() const { return outSize_; }

private:
    std::unique_ptr<Logger>                    log_;
    std::unique_ptr<nvinfer1::IRuntime>        runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>     engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<std::string>                   tensorNames_;
    std::string                                inName_, outName_;
    int64_t                                    inSize_{0}, outSize_{0};
    void*                                      dIn_{nullptr}, *dOut_{nullptr};
    cudaStream_t                               stream_{};
};

// -------------------------------- main --------------------------------
int main(int argc, char** argv) {
    try {
        // GPU check
        int devCount = 0;
        cudaGetDeviceCount(&devCount);
        std::cout << "CUDA device count: " << devCount << "\n";
        if (devCount > 0) {
            int dev = 0; cudaGetDevice(&dev);
            cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
            std::cout << "Using GPU #" << dev << ": " << prop.name
                      << " (Compute " << prop.major << "." << prop.minor << ")\n";
        }

        // parse args
        cxxopts::Options opt("vae_infer", "VAE inference (reconstruction MSE)");
        opt.add_options()
            ("e,engine",     "TensorRT engine (.plan)",   cxxopts::value<std::string>())
            ("d,data",       "Image directory",           cxxopts::value<std::string>())
            ("i,image_size", "Resize dimension",          cxxopts::value<int>())
            ("t,threshold",  "MSE threshold",             cxxopts::value<float>())
            ("h,help",       "Print help");
        auto a = opt.parse(argc, argv);
        if (a.count("help") || !a.count("engine") || !a.count("data")
            || !a.count("image_size") || !a.count("threshold")) {
            std::cout << opt.help() << std::endl;
            return 0;
        }

        const std::string engine_path = a["engine"].as<std::string>();
        const std::string data_dir    = a["data"].as<std::string>();
        const int         IMG_SIZE    = a["image_size"].as<int>();
        const float       THRESH      = a["threshold"].as<float>();

        // init context
        TrtCtx trt(engine_path);
        size_t IN  = trt.inputSize();   // now available
        size_t OUT = trt.outputSize();

        std::vector<float> recon(OUT);

        // gather images
        std::vector<fs::path> images;
        for (auto& p : fs::directory_iterator(data_dir))
            if (p.is_regular_file()) images.emplace_back(p);
        std::sort(images.begin(), images.end());

        // start
        std::cout << "=== Anomaly Inference Start (count: " << images.size() << ") ===\n";
        std::cout << "NOTE: raw scores are used; adjust threshold accordingly.\n";

        // CSV
        std::ofstream csv("result_vae.csv");
        csv << "filename,mse,decision\n";

        // timing
        Timer t_all; t_all.tic();
        double io_ms=0, prep_ms=0, infer_ms=0;
        Timer t;

        // main loop
        for (auto& f : images) {
            // I/O
            t.tic();
            cv::Mat img = cv::imread(f.string());
            io_ms += t.toc();
            if (img.empty()) continue;

            // preprocess
            t.tic();
            cv::resize(img, img, {IMG_SIZE, IMG_SIZE});
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            img.convertTo(img, CV_32FC3, 1.f/255);
            static const cv::Scalar MEAN(0.485,0.456,0.406),
                                     STD(0.229,0.224,0.225);
            img = (img - MEAN) / STD;
            std::vector<cv::Mat> ch(3); cv::split(img, ch);

            // input buffer sized by IN
            std::vector<float> input(IN);
            size_t plane = IMG_SIZE * IMG_SIZE;
            // assume IN == 3*plane
            for (int c = 0; c < 3; ++c) {
                std::memcpy(input.data() + c*plane,
                            ch[c].data, plane * sizeof(float));
            }
            prep_ms += t.toc();

            // inference
            t.tic();
            trt.infer(input.data(), recon.data());
            infer_ms += t.toc();

            // compute MSE over IN elements
            double sumsq = 0;
            for (size_t i = 0; i < IN; ++i) {
                double diff = input[i] - recon[i];
                sumsq += diff * diff;
            }
            double mse = sumsq / static_cast<double>(IN);

            std::string dec = mse > THRESH ? "Anomaly" : "Normal";
            csv << f.filename().string() << ',' << mse << ',' << dec << '\n';
        }
        csv.close();

        // summary + announce end
        double total_ms = t_all.toc();
        double avg_inf  = images.empty() ? 0.0 : infer_ms / images.size();
        std::cout
          << "I/O time       : " << io_ms    << " ms\n"
          << "Preprocess time: " << prep_ms  << " ms\n"
          << "Inference time : " << infer_ms << " ms\n"
          << "Avg infer/img  : " << avg_inf  << " ms\n"
          << "Results saved to result_vae.csv\n"
          << "=== Anomaly Inference End (" << total_ms << " ms) ===\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
