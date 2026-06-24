#include "tensorrt_infer.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <numeric>
using namespace std;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (false)

std::vector<char> loadCacheFromFile(const std::string& cacheFile) {
    std::ifstream file(cacheFile, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cout << "Cache file not found. A new one will be created." << std::endl;
        return {};
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
        std::cout << "Loaded " << size << " bytes from cache file: " << cacheFile << std::endl;
    }
    else {
        std::cerr << "Failed to read cache file." << std::endl;
        return {};
    }
    return buffer;
}
TensorRTEngine::TensorRTEngine(const string& onnx_path, size_t maxWorkspaceBytes, int device)
{
    string cache_path = onnx_path.substr(0, onnx_path.length() - 5) + std::to_string(MAX_BATCH_SIZE) + ".cache";
    std::ifstream cache_file(cache_path, std::ios::binary);
    static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    IBuilder* builder = createInferBuilder(m_logger);
    IBuilderConfig* config = builder->createBuilderConfig();
    INetworkDefinition* network = builder->createNetworkV2(EXPLICIT_BATCH);
    std::vector<char> cache_data = loadCacheFromFile(cache_path);
    IParser* parser = createParser(*network, m_logger);
    ITimingCache* cache = config->createTimingCache(
        cache_data.empty() ? nullptr : cache_data.data(),
        cache_data.size()
    );
    if (!cache)
        std::cerr << "Failed to create timing cache." << std::endl;
    else if (!config->setTimingCache(*cache, false))
        std::cerr << "Warning: Failed to set timing cache. Proceeding without it." << std::endl;
    else std::cout << "Timing cache attached to config." << std::endl;

    bool parser_success = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    for (int i = 0; i < 2; ++i)
    {
        Dims dim = network->getInput(i)->getDimensions();
        if (dim.d[0] == -1)  // -1 means it is a dynamic model
        {
            const char* name = network->getInput(i)->getName();
            dim.d[0] = 1;
            profile->setDimensions(name, OptProfileSelector::kMIN, dim);
            dim.d[0] = MAX_BATCH_SIZE;
            profile->setDimensions(name, OptProfileSelector::kOPT, dim);
            dim.d[0] = MAX_BATCH_SIZE;
            profile->setDimensions(name, OptProfileSelector::kMAX, dim);
        }
    }

    config->addOptimizationProfile(profile);
    config->setPreviewFeature(PreviewFeature::kPROFILE_SHARING_0806, true);
    config->setMaxWorkspaceSize(maxWorkspaceBytes);
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else {
        cout << "Platform does not support fast FP16, engine will still try but may run in FP32." << endl;
    }
    IHostMemory* serialized = builder->buildSerializedNetwork(*network, *config);
    CHECK(cudaSetDevice(0));
    runtime = createInferRuntime(m_logger);
    engine = runtime->deserializeCudaEngine(serialized->data(), serialized->size());
    if (cache)
    {
        IHostMemory* serializedCache = cache->serialize();
        std::ofstream file(cache_path, std::ios::binary);
        file.write(static_cast<const char*>(serializedCache->data()), serializedCache->size());
        if (file.good())
            std::cout << "Saved " << serializedCache->size() << " bytes to cache file: " << cache_path << std::endl;
        else std::cout << "Failed to write cache file." << std::endl;
        delete serializedCache;
    }
    delete serialized;
    delete cache;
    delete parser;
    delete config;
    delete network;
    delete builder;
}

TensorRTEngine::~TensorRTEngine()
{
    delete engine;
	delete runtime;
}
