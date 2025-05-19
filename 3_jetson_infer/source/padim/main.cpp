// ---------------------------------------------------------------------------
// main.cpp - PaDiM inference (Jetson Orin) with TensorRT 10.3
// ---------------------------------------------------------------------------
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "cxxopts.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>
#include <numeric>
#include <cstring>

namespace fs  = std::filesystem;
using Clock   = std::chrono::high_resolution_clock;
using msec_d  = std::chrono::duration<double, std::milli>;

// ---------------- Timer helper ----------------
struct Timer {
    Clock::time_point t0;
    void tic()         { t0 = Clock::now(); }
    double toc() const { return std::chrono::duration_cast<msec_d>(Clock::now() - t0).count(); }
};

// ----------------CUDA error check ----------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t status = (call);                                        \
        if (status != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status)       \
                      << std::endl;                                         \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// ---------------- TensorRT logger ----------------
class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << '\n';
    }
};

// ---------------- TensorRT context ----------------
class TrtCtx {
public:
    explicit TrtCtx(const std::string& plan) {
        log_ = std::make_unique<Logger>();
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

        // input tensor info
        inName_  = engine_->getIOTensorName(0);
        auto inShape = engine_->getTensorShape(inName_.c_str());
        inSize_ = std::accumulate(inShape.d, inShape.d + inShape.nbDims, 1LL,
                                  std::multiplies<int64_t>());

        // find the (single) output tensor
        int nbIO = engine_->getNbIOTensors();
        size_t maxElems = 0;
        for (int i = 0; i < nbIO; ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (!name || std::string(name) == inName_) continue;
            auto shape = engine_->getTensorShape(name);
            size_t elems = std::accumulate(shape.d, shape.d + shape.nbDims, 1LL,
                                           std::multiplies<int64_t>());
            if (elems > maxElems) { maxElems = elems; outName_ = name; }
        }
        outSize_ = maxElems;

        // collect binding names
        for (int i = 0; i < nbIO; ++i)
            if (auto n = engine_->getIOTensorName(i))
                tensorNames_.emplace_back(n);

        CUDA_CHECK(cudaMalloc(&dIn_,  inSize_  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut_, outSize_ * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~TrtCtx() {
        cudaStreamDestroy(stream_);
        cudaFree(dIn_);
        cudaFree(dOut_);
    }

    void infer(const float* hIn, float* hOut) {
        CUDA_CHECK(cudaMemcpyAsync(dIn_, hIn, inSize_*sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        for (auto& n : tensorNames_) {
            if (n == inName_)       context_->setTensorAddress(n.c_str(), dIn_);
            else if (n == outName_) context_->setTensorAddress(n.c_str(), dOut_);
            else                    context_->setTensorAddress(n.c_str(), dIn_);
        }
        context_->enqueueV3(stream_);
        CUDA_CHECK(cudaMemcpyAsync(hOut, dOut_, outSize_*sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    size_t outputSize() const { return outSize_; }

private:
    std::unique_ptr<Logger> log_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<std::string> tensorNames_;
    std::string inName_, outName_;
    int64_t inSize_{0}, outSize_{0};
    void* dIn_{nullptr};
    void* dOut_{nullptr};
    cudaStream_t stream_{};
};

// -------------------------------- main --------------------------------─
int main(int argc, char** argv) {
    try {
        // Verify GPU is available
        int devCount = 0;
        cudaError_t err = cudaGetDeviceCount(&devCount);
        std::cout << "CUDA device count: " << devCount << "\n";
        if (devCount > 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, dev);
            std::cout << "Using GPU #" << dev << ": " << prop.name
                    << " (Compute " << prop.major << "." << prop.minor << ")\n";
        }

        cxxopts::Options opt("padim",
            "PaDiM inference (embedded raw_score) with per-stage timing");
        opt.add_options()
            ("e,engine",    "TensorRT engine (.plan)",   cxxopts::value<std::string>())
            ("d,data",      "image directory",           cxxopts::value<std::string>())
            ("i,image_size","resize size (train size)", cxxopts::value<int>())
            ("t,threshold", "anomaly threshold",         cxxopts::value<float>())
            ("h,help",      "print help");
        auto a = opt.parse(argc, argv);
        if (a.count("help") || !a.count("engine") || !a.count("data")
         || !a.count("image_size") || !a.count("threshold")) {
            std::cout << opt.help() << std::endl;
            return 0;
        }

        const std::string engine_path = a["engine"].as<std::string>();
        const std::string data_dir    = a["data"].as<std::string>();
        const int IMG_SIZE            = a["image_size"].as<int>();
        const float THRESH            = a["threshold"].as<float>();

        // ---- load engine ----
        TrtCtx trt(engine_path);
        const size_t OUT = trt.outputSize();
        if (OUT != 1)
            throw std::runtime_error("Unexpected output size, expect 1");

        std::vector<float> raw_score(OUT);

        // gather image paths
        std::vector<fs::path> images;
        for (auto& p : fs::directory_iterator(data_dir))
            if (p.is_regular_file()) images.emplace_back(p);
        std::sort(images.begin(), images.end());

        std::cout << "=== Anomaly Inference Start (count: "
        << images.size() << ") ===\n"
        << "NOTE: raw scores are used; adjust threshold accordingly.\n";

        std::ofstream csv("result_padim.csv");
        csv << "filename,raw_score,decision\n";

        // timers
        Timer t_all; t_all.tic();
        double io_ms=0, prep_ms=0, infer_ms=0;
        Timer t;

        for (auto& fp : images) {
            // I/O
            t.tic();
            cv::Mat img = cv::imread(fp.string());
            io_ms += t.toc();
            if (img.empty()) { std::cerr<<"Skip "<<fp<<'\n'; continue; }

            // preprocess
            t.tic();
            cv::resize(img, img, {IMG_SIZE, IMG_SIZE});
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            img.convertTo(img, CV_32FC3, 1.f/255);
            static const cv::Scalar MEAN(0.485,0.456,0.406),
                                     STD (0.229,0.224,0.225);
            img = (img - MEAN) / STD;
            std::vector<cv::Mat> ch;
            cv::split(img, ch);
            std::vector<float> input(IMG_SIZE*IMG_SIZE*3);
            size_t total = static_cast<size_t>(IMG_SIZE) * IMG_SIZE;
            for (int c = 0; c < 3; ++c)
                std::memcpy(input.data() + c*total,
                            ch[c].data, total * sizeof(float));
            prep_ms += t.toc();

            // inference
            t.tic();
            trt.infer(input.data(), raw_score.data());
            infer_ms += t.toc();

            float raw = raw_score[0];
            std::string dec = (raw > THRESH ? "Anomaly" : "Normal");
            csv << fp.filename().string() << ',' << raw << ',' << dec << '\n';
        }
        csv.close();

        double total_ms = t_all.toc();
        std::cout << "I/O time         : " << io_ms    << " ms\n"
                  << "Preprocess time  : " << prep_ms  << " ms\n"
                  << "Inference time   : " << infer_ms << " ms\n"
                  << "Avg total/image  : "
                  << (images.empty() ? 0 : total_ms / images.size()) << " ms\n"
                  << "result_padim.csv saved\n"
                  << "=== Anomaly Inference End (" << total_ms << " ms) ===\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
