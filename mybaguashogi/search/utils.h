#pragma once

//#include <torch/script.h>  // One-stop header.
//#include <torch/torch.h>

#include <future>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <fstream>
#include <vector>
#include <execution>
#include <algorithm>

#include "tensorrt_infer.h"

#include "../game/board.h"
#include "../game/gamelogic.h"
#include "zlib.h"

extern std::mutex cout_mutex;
//using namespace torch;
//std::pair<Tensor, Tensor> boardToState(const Board& board);
constexpr size_t BINARY_SIZE = 55 * 9 * 9;
constexpr size_t OVERALL_SIZE = 15;
constexpr size_t POLICY_SIZE = 4790;


extern size_t MAX_BATCH_SIZE;

void boardToStateRaw(const Board& board, std::vector<int8_t>& bin, std::vector<float>& overall);
void boardToSaveRaw(const Board& board, std::vector<Piece>& color, std::vector<float>& overall);
template<class It, class T>
void softmax(It begin, T& probs, float temperature=1.f)
{
	probs.resize(POLICY_SIZE);
	std::transform(std::execution::par_unseq, begin, begin + POLICY_SIZE, probs.begin(), [temperature](const float& x) { return exp(x / temperature); });
	float sum = std::reduce(std::execution::par_unseq, probs.begin(), probs.end());
	//{
	//	std::lock_guard<std::mutex> lk(cout_mutex);
	//	cout << "sum:" << sum << endl;
	//}
	std::transform(std::execution::par_unseq, probs.begin(), probs.end(), probs.begin(), [sum](const float& x) { return x / sum; });
}
template<class T>
void append(std::vector<T>& a, std::vector<T>& x)
{
	size_t old_size = a.size();
	a.resize(a.size() + x.size());
	std::copy(std::execution::par_unseq, x.begin(), x.end(), a.begin() + old_size);
}
template<typename T>
void save_vector_zlib(const std::string& filename, const std::vector<T>& data) {
	uLong src_len = data.size() * sizeof(T);
	uLong dest_len = compressBound(src_len);
	std::vector<Bytef> compressed(dest_len);

	int res = compress2(compressed.data(), &dest_len, reinterpret_cast<const Bytef*>(data.data()), src_len, Z_BEST_COMPRESSION);
	if (res != Z_OK)
		throw std::runtime_error("zlib compress failed");

	std::ofstream out(filename, std::ios::binary);
	uint64_t orig_size = data.size();
	out.write(reinterpret_cast<const char*>(compressed.data()), dest_len);   // 保存压缩数据
	out.close();
	cout << "文件已保存到: " << filename << endl;
}
class IndexIterator
{
	size_t index;
public:
	using iterator_category = std::forward_iterator_tag;
	using value_type = size_t;
	using difference_type = std::ptrdiff_t;
	using pointer = const size_t*;
	using reference = const size_t&;

	IndexIterator(size_t index = 0) :index(index) {}
	value_type operator*() const { return index; }
	IndexIterator& operator++() { ++index; return *this; }
	bool operator!=(const IndexIterator& other) const { return index != other.index; }
};
//class NeuralNetwork {
// public:
//  using return_type = std::vector<std::vector<double>>;
//
//  NeuralNetwork(std::string model_path, bool use_gpu, unsigned int batch_size);
//  ~NeuralNetwork();
//
//  std::future<return_type> commit(Board* game);  // commit task to queue
//  void set_batch_size(unsigned int batch_size) {    // set batch_size
//    this->batch_size = batch_size;
//  };
//
// private:
//  using task_type = std::pair<torch::Tensor, std::promise<return_type>>;
//
//  void infer();  // infer
//
//  std::unique_ptr<std::thread> loop;  // call infer in loop
//  bool running;                       // is running
//
//  std::queue<task_type> tasks;  // tasks queue
//  std::mutex lock;              // lock for tasks queue
//  std::condition_variable cv;   // condition variable for tasks queue
//
//  std::shared_ptr<torch::jit::script::Module> module;  // torch module
//  unsigned int batch_size;                             // batch size
//  bool use_gpu;                                        // use gpu
//};
