#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <execution>
#include <NvInfer.h>
#include <iostream>
#include <chrono>
#include "utils.h"
#include "tensorrt_infer.h"
using namespace std::chrono_literals;
#define cudaMemcpy __CUDART_API_PTDS(cudaMemcpy)
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

using namespace nvinfer1;
using std::vector;
class Buffer
{
public:
	friend class bufferpool;
	using inputType = float;
	using resultType = std::pair<std::vector<inputType>, std::vector<inputType>>;
	static std::atomic_uint waiting_num;
	Buffer(IExecutionContext* context) : context(context)
	{
		if (!context) throw std::runtime_error("failed to create TensorRT execution context");
		CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
		CHECK(cudaMalloc(buffers + 0, MAX_BATCH_SIZE * BINARY_SIZE * sizeof(inputType)));
		CHECK(cudaMalloc(buffers + 1, MAX_BATCH_SIZE * OVERALL_SIZE * sizeof(inputType)));
		CHECK(cudaMalloc(buffers + 2, MAX_BATCH_SIZE * POLICY_SIZE * sizeof(float)));
		CHECK(cudaMalloc(buffers + 3, MAX_BATCH_SIZE * sizeof(float)));

		context->setTensorAddress("binary", buffers[0]);
		context->setTensorAddress("overall", buffers[1]);
		context->setTensorAddress("policy", buffers[2]);
		context->setTensorAddress("value", buffers[3]);
	}
	Buffer(const Buffer&) = delete;
	Buffer& operator=(const Buffer&) = delete;
	Buffer(Buffer&& other) = delete;
	~Buffer() {
		for (int i = 0; i < 4; ++i) if (buffers[i]) CHECK(cudaFree(buffers[i]));
		if (stream) CHECK(cudaStreamDestroy(stream));
		delete context;
	}
	operator void** () { return buffers; }
	operator const void* const* () const { return buffers; }
	resultType infer(const vector<inputType>& bin_flat, const vector<inputType>& overall_flat)
	{
		if (overall_flat.size() % OVERALL_SIZE != 0)throw std::runtime_error("overall_flat size must be a multiple of OVERALL_SIZE");
		//vector<float> bin(bin_flat.size());
		int batch_size = overall_flat.size() / OVERALL_SIZE;
		size_t policy_elems = batch_size * POLICY_SIZE;
		size_t value_elems = batch_size;
		vector<inputType> policy_out(policy_elems), value_out(value_elems);
		CHECK(cudaMemcpyAsync(buffers[0], bin_flat.data(), bin_flat.size() * sizeof(inputType), cudaMemcpyHostToDevice, stream));
		CHECK(cudaMemcpyAsync(buffers[1], overall_flat.data(), overall_flat.size() * sizeof(inputType), cudaMemcpyHostToDevice, stream));
		++waiting_num;
		if (!context->setInputShape("binary", Dims4{ batch_size, 55, 9, 9 }))
			throw std::runtime_error("failed to set binary input shape");
		if (!context->setInputShape("overall", Dims2{ batch_size, 15 }))
			throw std::runtime_error("failed to set overall input shape");
		if (!context->enqueueV3(stream))
			throw std::runtime_error("TensorRT enqueue failed");
		CHECK(cudaMemcpyAsync(policy_out.data(), buffers[2], policy_elems * sizeof(inputType), cudaMemcpyDeviceToHost, stream));
		CHECK(cudaMemcpyAsync(value_out.data(), buffers[3], value_elems * sizeof(inputType), cudaMemcpyDeviceToHost, stream));
		CHECK(cudaStreamSynchronize(stream));
		--waiting_num;
		in_use = false;
		return { move(policy_out), move(value_out) };
	}
private:
	void* buffers[4];
	IExecutionContext* context;
	cudaStream_t stream = nullptr;
	bool in_use = false;
};
class bufferpool
{
public:
	using resultType = std::pair<std::vector<float>, std::vector<float>>;
	bufferpool(TensorRTEngine* engine, size_t max_size = 5) : engine(engine), max_size(max_size) {}
	~bufferpool() = default;
	size_t size() const { return buffers.size(); };
	bool empty() const { return buffers.empty(); }
	Buffer* get_buffer()
	{
		std::unique_lock<std::mutex> lk(lock, std::defer_lock);
		int index = -1;
		lk.lock();
		for (int i = 0; index == -1 && i < buffers.size(); ++i)
			if (!buffers[i]->in_use)
				index = i;
		if (index == -1 && buffers.size() < max_size)
		{
			index = buffers.size();
			buffers.emplace_back(std::make_unique<Buffer>(engine->createContext()));
		}
		if (index == -1)return nullptr;
		buffers[index]->in_use = true;
		return buffers[index].get();
	}
private:
	std::vector<std::unique_ptr<Buffer>> buffers;
	size_t max_size;
	TensorRTEngine* engine;
	std::mutex lock;
};

