#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <execution>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "utils.h"
#include "../samples/common/logger.h"
using namespace sample;
using namespace nvinfer1;
using namespace nvonnxparser;

class TensorRTEngine {
public:
    // onnx_path: ONNX 模型路径
    // maxWorkspaceBytes: TensorRT 最大 workspace（默认为 1GB）
    // device: CUDA 设备 id
    TensorRTEngine(const std::string& onnx_path, size_t maxWorkspaceBytes = 1ULL << 30, int device = 0);
    ~TensorRTEngine();
    IExecutionContext* createContext() const { return engine->createExecutionContext(); }
private:
    Logger m_logger;
    ICudaEngine* engine;
	IExecutionContext* context;
    IRuntime* runtime;
};