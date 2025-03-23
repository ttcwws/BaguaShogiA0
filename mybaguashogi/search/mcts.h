#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <blockingconcurrentqueue.h>

#include "../game/board.h"
#include "../threadpool2.h"
#include "libtorch.h"
//#include <libtorch.h>

class TreeNode {
public:
	// friend class can access private variables
	friend class MCTS;
	friend int main(int argc, char* argv[]);
	TreeNode();
	TreeNode(const TreeNode& node);
	TreeNode(TreeNode* parent, float p_sa);

	TreeNode& operator=(const TreeNode& p);

	unsigned int select(double c_puct, double c_virtual_loss)const;
	unsigned int select_forced(double c_puct, double c_virtual_loss)const;
	//void expand(const std::vector<double>& action_priors);
	void expand(const std::vector<size_t>& legal_pols, const std::vector<float>& policy_probs);
	void backup(double leaf_value);
	unsigned get_n_force(unsigned sum_n_visited);
	unsigned get_visited()const;
	double get_value(double c_puct, double c_virtual_loss,
		unsigned int sum_n_visited) const;
	inline bool get_is_leaf() const { return this->is_leaf; }

private:
	// store tree
	TreeNode* parent;
	TreeNode** children;
	int* legals;
	int legal_size;
	bool is_leaf;
	std::mutex lock;

	std::atomic<unsigned int> n_visited;
	double p_sa;
	double q_sa;
	std::atomic<int> virtual_loss;
};
struct NetWorkParallelizer
{
	torch::jit::Module* model;
	std::mutex lock;
	bool clear;
	int batch_size;
	using result_type = std::pair<std::vector<float>, float>;
	Tensor t_bin, t_overall;
	moodycamel::BlockingConcurrentQueue<Tensor> tasks_bin, tasks_overall;
	moodycamel::ConcurrentQueue<std::thread> threads;
	moodycamel::BlockingConcurrentQueue<std::promise<result_type>> promises;
	std::vector<std::promise<result_type>> proms;
	std::unique_ptr<std::thread> loop, loop2, loop3;
	bool running;
	bool use_gpu;
	std::atomic_bool inferable, idle;
	NetWorkParallelizer(torch::jit::Module* model, int batch_size, bool use_gpu = true) :model(model), batch_size(batch_size), clear(false), running(true), use_gpu(use_gpu), inferable(false), idle(true),
		loop(std::make_unique<std::thread>([this] {
		s = clock();
		while (this->running)this->infer(); })),

		loop2(std::make_unique<std::thread>([this] {
		while (this->running)
		{
			std::thread th;
			while (threads.try_dequeue(th))th.join();
		}})),

		loop3(std::make_unique<std::thread>([this] {
		while (this->running)this->get_data(); })) {}
	~NetWorkParallelizer();
	std::condition_variable cv;
	std::future<result_type> pushState(Tensor bin, Tensor overall);
	void infer();
	void get_data();
	clock_t s;
};
struct NetWorkParallelizerLock
{
	torch::jit::Module* model;
	std::mutex lock;
	bool clear;
	int batch_size, max_size;
	using result_type = std::pair<std::vector<float>, float>;
	std::queue<Tensor> tasks_bin, tasks_overall;
	//Tensor t_bin, t_overall;
	//std::vector<std::promise<result_type>> proms;
	moodycamel::ConcurrentQueue<std::thread> threads;
	std::queue<std::promise<result_type>> promises;
	std::unique_ptr<std::thread> loop, loop2;
	bool running;
	bool use_gpu;
	//std::atomic_bool inferable;
	NetWorkParallelizerLock(torch::jit::Module* model, int batch_size, bool use_gpu = true) :model(model), max_size(batch_size), batch_size(batch_size), clear(false), running(true), use_gpu(use_gpu),
		loop(std::make_unique<std::thread>([this] {
		s = clock();
		while (this->running)this->infer(); })),

		loop2(std::make_unique<std::thread>([this] {
		while (this->running)
		{
			std::thread th;
			while (threads.try_dequeue(th))th.join();
		}}))

/*		loop3(std::make_unique<std::thread>([this] {
		while (this->running)this->get_data(); }))*/ {}
	~NetWorkParallelizerLock();
	std::condition_variable cv;
	std::future<result_type> pushState(Tensor bin, Tensor overall);
	void infer();
	//void get_data();
	//void infer(Tensor t_bin, Tensor t_overall,std::vector<std::promise<result_type>>& proms);
	clock_t s;
};
using NetWork = NetWorkParallelizer;
class MCTS {
public:
	friend class SelfGame;
	friend int main(int argc, char* argv[]);
	MCTS(NetWork* neural_network, unsigned int thread_num, double c_puct,
		unsigned int num_mcts_sims, double c_virtual_loss,
		unsigned int action_size, double alpha, bool forced, bool gpu, bool noise = true);
	std::vector<double> get_action_probs(Board* board, double temp = 1e-3);
	void get_action_visits(Board* board, std::vector<unsigned>& action, std::vector<unsigned>& action_pruned);
	void update_with_move(int last_move);

private:
	void simulate(std::shared_ptr<Board> game);
	static void tree_deleter(TreeNode* t);

	// variables
	std::unique_ptr<TreeNode, decltype(MCTS::tree_deleter)*> root;
	std::unique_ptr<std::threadpool> thread_pool;
	//NeuralNetwork* neural_network;
	NetWork* neural_network;

	unsigned int action_size;
	unsigned int num_mcts_sims;
	double c_puct;
	double c_virtual_loss;
	double alpha;
	bool use_gpu;
	bool forced;
	bool noise;
};
