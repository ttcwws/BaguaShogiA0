#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <concurrentqueue.h>
#include <execution>
#include <algorithm>

#include "../game/board.h"
#include "../search/thread_pool.h"
#include "utils.h"
#include "bufferpool.h"
#include "tensorrt_infer.h"
class TreeNode {
public:
	// friend class can access private variables
	friend class MCTS;
	friend int main(int argc, char* argv[]);
	TreeNode();
	TreeNode(const TreeNode& node);
	TreeNode(TreeNode* parent, unsigned action, float p_sa);

	TreeNode& operator=(const TreeNode& p);

	TreeNode* select(float c_puct, float c_fpu);
	TreeNode* select_forced(float c_puct);
	void trim();
	//void expand(const std::vector<double>& action_priors);
	void expand(const std::vector<size_t>& legal_pols, const std::vector<float>& policy_probs);
	void sort_children();
	void add_dirichlet_noise(float alpha, float epsilon = .25f, float t = 1.18f);
	void backup(float value);
	unsigned get_n_force(unsigned sum_n_visited);
	unsigned get_visited()const;
	float get_value() const;
	inline bool get_is_leaf() const { return this->is_leaf; }

private:
	// store tree
	TreeNode* parent;
	std::vector<std::shared_ptr<TreeNode>> children; // 改为vector
	std::vector<std::pair<size_t, float>> lazy_children; // action, p_sa
	bool is_leaf;
	int determine;
	float value_self;

	unsigned n_visited;
	float p_sa, p_ex;
	float value_sum;
	unsigned action;
};
struct NetWorkParallelizer
{
public:
	using result_type = std::pair<std::vector<float>, float>;
private:
	size_t max_thread_num = 8;
	static constexpr size_t NN_CACHE_SIZE = 1 << 12;
	static constexpr size_t NN_CACHE_MASK = NN_CACHE_SIZE - 1;
	static constexpr size_t NN_CACHE_MUTEX_POOL_SIZE = 1 << 8;
	static constexpr size_t NN_CACHE_MUTEX_POOL_MASK = NN_CACHE_MUTEX_POOL_SIZE - 1;
	struct CacheEntry
	{
		Hash128 hash;
		std::shared_ptr<const result_type> value;
		CacheEntry() : hash(), value(nullptr) {}
	};
public:
	TensorRTEngine* model;
	bufferpool bufpool;

	float temperature;
	std::vector<std::promise<result_type>> proms;
	moodycamel::ConcurrentQueue<std::tuple<std::promise<result_type>, std::vector<int8_t>, std::vector<float>>> tasks;
	ThreadPool thp;
	std::future<void> get_worker;
	std::atomic_bool running;
	std::vector<CacheEntry> nn_cache;
	std::vector<std::mutex> nn_cache_mutex_pool;
	std::atomic<uint64_t> nn_cache_hits;
	std::atomic<uint64_t> nn_cache_misses;
	bool use_gpu;
	NetWorkParallelizer(TensorRTEngine* model, float temperature, bool use_gpu = true)
		: model(model), running(true), nn_cache(NN_CACHE_SIZE), nn_cache_mutex_pool(NN_CACHE_MUTEX_POOL_SIZE), nn_cache_hits(0), nn_cache_misses(0), temperature(temperature), use_gpu(use_gpu), thp(max_thread_num), bufpool(model, max_thread_num)
	{
		get_worker = thp.commit(std::bind(&NetWorkParallelizer::get_data, this));
	}
	~NetWorkParallelizer();
	result_type predict(const Board& board);
	void get_data();
private:
	bool try_get_cache(const Hash128& hash, result_type& result);
	void put_cache(const Hash128& hash, const result_type& result);
	std::mutex lock;
};
using NetWork = NetWorkParallelizer;
class MCTS {
public:
	friend class SelfGame;
	friend int main(int argc, char* argv[]);
	MCTS(NetWork* neural_network, float c_puct,
		unsigned int num_mcts_sims,
		unsigned int action_size, float alpha, bool forced, bool gpu, bool noise = true);
	std::vector<double> get_action_probs(Board* board, double temp = 1e-3);
	void get_action_visits(const Board& board, std::vector<unsigned>& action, std::vector<unsigned>& action_pruned);
	void update_with_move(int last_move);

private:
	void simulate(Board game);

	// variables
	std::shared_ptr<TreeNode> root;
	std::unique_ptr<ThreadPool> thread_pool;
	//NeuralNetwork* neural_network;
	NetWork* neural_network;

	unsigned int action_size;
	unsigned int num_mcts_sims;
	float c_puct;
	float alpha;
	bool use_gpu;
	bool forced;
	bool noise;
};
