#include <cmath>
#include <cfloat>
#include <numeric>
#include <iostream>
#include "Windows.h"

#include "mcts.h"
using namespace std::chrono_literals;
void set_promises(Tensor probs, Tensor values, std::vector<std::promise<NetWork::result_type>>& proms)
{
	for (int i = 0; i < proms.size(); ++i)
	{
		float val = values[i].item<float>();
		proms[i].set_value(std::make_pair(
			std::vector<float>(static_cast<float*>(probs[i].data_ptr()), static_cast<float*>(probs[i].data_ptr()) + probs[i].numel()),
			val
		));
	}
}
NetWorkParallelizer::~NetWorkParallelizer()
{
	running = false;
	loop->join();
	loop2->join();
	loop3->join();
}
std::future<NetWorkParallelizer::result_type> NetWorkParallelizer::pushState(Tensor bin, Tensor overall)
{
	std::promise<result_type> promise;
	auto ret = promise.get_future();
	tasks_bin.enqueue(std::move(bin.unsqueeze(0)));
	tasks_overall.enqueue(std::move(overall.unsqueeze(0)));
	promises.enqueue(std::move(promise));
	return ret;
}
void NetWorkParallelizer::get_data()
{
	//if (inferable)return;
	TensorOptions device(use_gpu ? at::kCUDA : at::kCPU);
	NoGradGuard _no_grad;
	std::vector<Tensor> bin, overall;
	if (!proms.empty())return;
	std::unique_lock<std::mutex> _lock(lock);
	while (proms.size() < batch_size)
	{
		//std::unique_lock<std::mutex> _lock(lock);
		Tensor _bin, _overall;
		std::promise<result_type> _promise;
		if (promises.wait_dequeue_timed(_promise, 1ms))
		{
			if (!tasks_bin.try_dequeue(_bin) || !tasks_overall.try_dequeue(_overall))
			{
				cout << "PROMISED BUT NO TASK" << endl;
				throw "PROMISED BUT NO TASK";
			}
			proms.emplace_back(std::move(_promise));
			bin.emplace_back(_bin);
			overall.emplace_back(_overall);
			if (idle)break;
		}
		else break;
	}
	if (!proms.empty())
	{
		//cout << proms.size() << endl;
		t_bin = cat(bin).to(device);
		t_overall = cat(overall).to(device);
	}
}
void NetWorkParallelizer::infer()
{
	TensorOptions device(use_gpu ? at::kCUDA : at::kCPU);
	//if (!inferable)return;
	c10::ivalue::TupleElements result;
	Tensor t_bin, t_overall;
	if (model)
	{
		//s = clock();

		//cout << "part1.5:" << (clock() - s) / (double)CLOCKS_PER_SEC << endl;
		//s = clock();
		//cout << prom.size() << endl;
		if (proms.empty())return;
		std::vector<std::promise<result_type>> proms;
		{
			std::unique_lock<std::mutex> _lock(lock);
			t_bin = this->t_bin;
			t_overall = this->t_overall;
			proms.swap(this->proms);
		}
		//inferable = false;
		idle = false;
		//cout << proms.size() << endl;
		try
		{
			//cout << clock() - s << endl;
			result = model->forward(torch::jit::Stack({ t_bin,t_overall })).toTuple()->elements();
			auto probs = softmax(result[0].toTensor(), 1).cpu(), values = result[1].toTensor().cpu();
			idle = true;
			//cout << probs.requires_grad() << endl;
			//cout << "part1:" << (clock() - s) / (double)CLOCKS_PER_SEC << endl;
			//s = clock();
			threads.enqueue(std::thread(std::bind(set_promises, std::move(probs), std::move(values), std::move(proms))));
		}
		catch (const c10::Error& e)
		{
			cout << e.msg() << endl;
		}

	}
	else throw "EMPTY MODEL";

}
//lock version
NetWorkParallelizerLock::~NetWorkParallelizerLock()
{
	running = false;
	loop->join();
	loop2->join();
	//loop3->join();
}
std::future<NetWorkParallelizerLock::result_type> NetWorkParallelizerLock::pushState(Tensor bin, Tensor overall)
{
	std::promise<result_type> promise;
	auto ret = promise.get_future();
	{
		std::lock_guard<std::mutex> _lock(lock);
		tasks_bin.emplace(bin.unsqueeze(0));
		tasks_overall.emplace(overall.unsqueeze(0));
		promises.emplace(std::move(promise));
	}
	cv.notify_all();
	return ret;
}
void NetWorkParallelizerLock::infer()
{
	//if (inferable)return;
	NoGradGuard _no_grad;
	std::vector<Tensor> bin, overall;
	std::vector<std::promise<result_type>> proms;
	while (bin.size() < batch_size)
	{
		std::unique_lock<std::mutex> _lock(lock);
		if (cv.wait_for(_lock, 1ms, [this] {return !this->promises.empty(); }))
		{
			while (bin.size() < batch_size && !promises.empty())
			{
				proms.emplace_back(std::move(promises.front()));
				bin.emplace_back(tasks_bin.front());
				overall.emplace_back(tasks_overall.front());
				promises.pop();
				tasks_bin.pop();
				tasks_overall.pop();
				//if (bin.size() >= batch_size)
				//	batch_size = std::min((size_t)max_size, promises.size() + bin.size());
				//else if (promises.empty())batch_size = bin.size() * .98 + 1;
			}
			if (bin.size() < batch_size)continue;
		}
		break;
	}
	//NoGradGuard _no_grad;
	if (proms.empty())return;
	TensorOptions device(use_gpu ? at::kCUDA : at::kCPU);
	//cout << proms.size() << endl;
	if (model)
	{
		Tensor t_bin, t_overall;
		//s = clock();
		t_bin = cat(bin).to(device);
		t_overall = cat(overall).to(device);
		//std::vector<std::promise<result_type>> proms(std::move(this->proms));
		//inferable = false;
		c10::ivalue::TupleElements result;
		//cout << clock() - s << endl;
		result = model->forward(torch::jit::Stack({ t_bin,t_overall })).toTuple()->elements();
		auto probs = softmax(result[0].toTensor(), 1).cpu(), values = result[1].toTensor().cpu();
		threads.enqueue(std::thread(std::bind(&set_promises, std::move(probs), std::move(values), std::move(proms))));
		//cout << inferable << endl;
		//s = clock();
		//std::lock_guard<std::mutex> _lock(lock);
		//threads.emplace(std::bind(&NetWorkParallelizerLock::infer, this, std::move(t_bin), std::move(t_overall), std::move(proms)));
		//static int cnt1 = 0;
		//static clock_t cnt2 = 0;
		//if (rand() % 1000 == 0)
			//cout << prom.size() << endl;
		//cout << "part1:" << (clock() - s) / (double)CLOCKS_PER_SEC << endl;


		//s = clock();
		//cout << probs.requires_grad() << endl;
		//cnt1 += batch_size;
		//cnt2 += (clock() - s);
		//if (rand() % 400 == 0)
		//	cout << "part1:" << (double)cnt1 / cnt2 * CLOCKS_PER_SEC<< endl, cnt1 = cnt2 = 0;
		//s = clock();

	}
	else throw "EMPTY MODEL";
	//if (proms.size())
	//{
	//	t_bin = cat(bin);
	//	t_overall = cat(overall);
	//	inferable = true;
	//}
	//cv.notify_all();
}
//void NetWorkParallelizerLock::infer()
//{
//	if (!inferable)return;
//	NoGradGuard _no_grad;
//	TensorOptions device(use_gpu ? at::kCUDA : at::kCPU);
//	//cout << proms.size() << endl;
//	if (model)
//	{
//		//s = clock();
//		Tensor t_bin = this->t_bin.to(device), t_overall = this->t_overall.to(device);
//		std::vector<std::promise<result_type>> proms(std::move(this->proms));
//		inferable = false;
//		c10::ivalue::TupleElements result;
//		//cout << clock() - s << endl;
//		result = model->forward(torch::jit::Stack({ t_bin,t_overall })).toTuple()->elements();
//		auto probs = softmax(result[0].toTensor(), 1).cpu(), values = result[1].toTensor().cpu();
//		threads.enqueue(std::thread(std::bind(&set_promises, std::move(probs), std::move(values), std::move(proms))));
//		//cout << inferable << endl;
//		//s = clock();
//		//std::lock_guard<std::mutex> _lock(lock);
//		//threads.emplace(std::bind(&NetWorkParallelizerLock::infer, this, std::move(t_bin), std::move(t_overall), std::move(proms)));
//		//static int cnt1 = 0;
//		//static clock_t cnt2 = 0;
//		//if (rand() % 1000 == 0)
//			//cout << prom.size() << endl;
//		//cout << "part1:" << (clock() - s) / (double)CLOCKS_PER_SEC << endl;
//
//
//		//s = clock();
//		//cout << probs.requires_grad() << endl;
//		//cnt1 += batch_size;
//		//cnt2 += (clock() - s);
//		//if (rand() % 400 == 0)
//		//	cout << "part1:" << (double)cnt1 / cnt2 * CLOCKS_PER_SEC<< endl, cnt1 = cnt2 = 0;
//		//s = clock();
//
//	}
//	else throw "EMPTY MODEL";
//}

// TreeNode
TreeNode::TreeNode()
	: parent(nullptr),
	is_leaf(true),
	virtual_loss(0),
	legal_size(0),
	n_visited(0),
	p_sa(0),
	q_sa(0) {}

TreeNode::TreeNode(TreeNode* parent, float p_sa)
	: parent(parent),
	//children(action_size, nullptr),
	is_leaf(true),
	virtual_loss(0),
	legal_size(0),
	n_visited(0),
	q_sa(0),
	p_sa(p_sa) {}

TreeNode::TreeNode(
	const TreeNode& node) {  // because automic<>, define copy function
	// struct
	this->parent = node.parent;
	this->children = node.children;
	this->is_leaf = node.is_leaf;

	this->n_visited.store(node.n_visited.load());
	this->p_sa = node.p_sa;
	this->q_sa = node.q_sa;

	this->virtual_loss.store(node.virtual_loss.load());
}

TreeNode& TreeNode::operator=(const TreeNode& node) {
	if (this == &node) {
		return *this;
	}

	// struct
	this->parent = node.parent;
	this->children = node.children;
	this->is_leaf = node.is_leaf;

	this->n_visited.store(node.n_visited.load());
	this->p_sa = node.p_sa;
	this->q_sa = node.q_sa;
	this->virtual_loss.store(node.virtual_loss.load());

	return *this;
}

unsigned int TreeNode::select(double c_puct, double c_virtual_loss) const {
	//throw '?';
	unsigned int best_move = 0;
	{
		// get lock
		//std::lock_guard<std::mutex> lock(this->lock);
		double best_value = -DBL_MAX;
		TreeNode* best_node = nullptr;
		//std::vector<size_t> v;
		//v.reserve(children.size());
		//for (const auto& [i, child] : children)
		//	v.push_back(i);
		//std::shuffle(v.begin(), v.end(), std::mt19937_64(std::random_device()()));
		for (int i = 0; i < legal_size; ++i)
		{
			const auto& child = children[i];
			// empty node
			if (!child)continue;

			unsigned int sum_n_visited = this->n_visited.load();
			double cur_value =
				child->get_value(c_puct, c_virtual_loss, sum_n_visited);
			//if (i == 31 || i == 440)
			//	cout << i << ' ' << child->q_sa << endl, getchar();
			//if (children[i]->virtual_loss)
			//{
			//	cout << "before:";
			//	cout << cur_value << endl;
			//	cout << "after:";
			//	++children[i]->virtual_loss;
			//	cout << children[i]->get_value(c_puct, c_virtual_loss, sum_n_visited);
			//	--children[i]->virtual_loss;
			//}
			if (cur_value > best_value) {
				//if (best_node)--best_node->virtual_loss;
				best_value = cur_value;
				best_move = i;
				best_node = child;
				// add vitural loss
			}
		}

		//cout << best_node->get_value(c_puct, c_virtual_loss, this->n_visited.load() + 1) << endl;
		++best_node->virtual_loss;
	}
	return best_move;
}
unsigned TreeNode::get_n_force(unsigned sum_n_visited)
{
	return (unsigned)sqrt(1.8 * p_sa * sum_n_visited);
}
unsigned TreeNode::get_visited() const
{
	return n_visited;
}
unsigned int TreeNode::select_forced(double c_puct, double c_virtual_loss) const {
	//throw '?';
	unsigned int best_move = 0;

	double best_value = -DBL_MAX;
	TreeNode* best_node = nullptr;
	/*std::vector<size_t> v;
	v.reserve(children.size());
	for (const auto& [i, child] : children)
		v.push_back(i);
	std::shuffle(v.begin(), v.end(), std::mt19937_64(std::random_device()()));*/
	//std::sort(v.begin(), v.end(), [this](const std::pair<size_t, unsigned>& a, const std::pair<size_t, unsigned>& b) {return a.second < b.second; });
	for (int i = 0; i < legal_size; ++i)
	{
		const auto& child = children[i];
		// empty node
		if (!legals[i])continue;
		unsigned int sum_n_visited = this->n_visited.load();
		//if (legals[i] == 49)
			//cout << legals[i] << ' ' << child->get_value(c_puct, c_virtual_loss, sum_n_visited)<<" q="<< child->q_sa<< " p=" <<child->p_sa<<" v="<<child->n_visited<< " V="<< sum_n_visited<<" forced="<< child->get_n_force(sum_n_visited) <<endl;
		if (child->n_visited + child->virtual_loss < child->get_n_force(sum_n_visited))
		{
			//if (best_node)--best_node->virtual_loss;
			best_move = i;
			best_node = child;
			//++best_node->virtual_loss;
			break;
		}
		//cout << child->p_sa << ' ' << sum_n_visited << ' ' << child->get_n_force(sum_n_visited) << endl;
		double cur_value =
			child->get_value(c_puct, c_virtual_loss, sum_n_visited);
		if (cur_value > best_value) {
			//if (best_node)--best_node->virtual_loss;
			best_value = cur_value;
			best_move = i;
			best_node = child;
			//++best_node->virtual_loss;
			// add vitural loss
		}
	}
	++best_node->virtual_loss;
	return best_move;
}

void TreeNode::expand(const std::vector<size_t>& legal_pols, const std::vector<float>& policy_probs)
{
	{
		// get lock
		std::lock_guard<std::mutex> lock(this->lock);
		if (is_leaf) {
			int* index = new int[legal_size = legal_pols.size()];
			legals = new int[legal_size];
			std::iota(index, index + legal_size, 0);
			std::shuffle(index, index + legal_size, std::mt19937_64(std::random_device()()));
			children = new TreeNode * [legal_size];
			for (int i = 0; i < legal_size; ++i)
			{
				legals[i] = legal_pols[index[i]];
				children[i] = new TreeNode(this, policy_probs[index[i]]);
			}
			// not leaf
			is_leaf = false;
			delete[] index;
		}
		//cout << children.size() << endl;
	}
}
//void TreeNode::expand(const std::vector<double>& action_priors) {
//	{
//		// get lock
//		std::lock_guard<std::mutex> lock(this->lock);
//		if (this->is_leaf) {
//			unsigned int action_size = this->children.size();
//
//			for (unsigned int i = 0; i < action_size; i++) {
//				// illegal action
//				if (abs(action_priors[i] - 0) < DBL_EPSILON) {
//					continue;
//				}
//				this->children[i] = new TreeNode(this, action_priors[i], action_size);
//			}
//
//			// not leaf
//			this->is_leaf = false;
//		}
//	}
//}

void TreeNode::backup(double value) {
	// If it is not root, this node's parent should be updated first
	if (this->parent != nullptr) {
		this->parent->backup(-value);
	}

	// remove vitural loss
	--this->virtual_loss;

	// update n_visited
	unsigned int n_visited = this->n_visited.load();
	this->n_visited++;
	// update q_sa
	{
		std::lock_guard<std::mutex> lock(this->lock);
		this->q_sa = (n_visited * this->q_sa + value) / (n_visited + 1);
	}
}

double TreeNode::get_value(double c_puct, double c_virtual_loss,
	unsigned int sum_n_visited) const {
	// u
	auto n_visited = this->n_visited.load();
	auto virtual_visited = this->virtual_loss.load() * c_virtual_loss;
	//double u = (c_puct * this->p_sa * sqrt(sum_n_visited) / (n_visited + 1));
	double u = (c_puct * this->p_sa * sqrt(sum_n_visited + virtual_visited) / (n_visited + virtual_visited + 1));

	// virtual loss
	double virtual_loss = 0;
	//double virtual_loss = c_virtual_loss * this->virtual_loss.load();
	//double virtual_loss = c_virtual_loss * sqrt(n_visited + this->virtual_loss.load()) / (this->virtual_loss.load() + 1);
	// int n_visited_with_loss = n_visited - virtual_loss;

	//if (n_visited <= 0) {
		//return u;
	//}
	//return u + (this->q_sa * n_visited - virtual_loss) / n_visited;
	return u + this->q_sa + virtual_loss;

}

// MCTS
MCTS::MCTS(NetWork* neural_network, unsigned int thread_num, double c_puct,
	unsigned int num_mcts_sims, double c_virtual_loss,
	unsigned int action_size, double alpha, bool forced, bool gpu, bool noise)
	: neural_network(neural_network),
	thread_pool(new std::threadpool(thread_num)),
	c_puct(c_puct),
	num_mcts_sims(num_mcts_sims),
	c_virtual_loss(c_virtual_loss),
	action_size(action_size),
	alpha(alpha),
	use_gpu(gpu),
	forced(forced),
	noise(noise),
	root(new TreeNode(nullptr, 1), MCTS::tree_deleter) {}
void MCTS::update_with_move(int last_action) {
	auto old_root = this->root.get();
	// reuse the child tree
	int index = std::find(old_root->legals, old_root->legals + old_root->legal_size, last_action) - old_root->legals;
	if (index != old_root->legal_size) {
		// unlink
		TreeNode* new_node = old_root->children[index];
		old_root->children[index] = nullptr;
		old_root->legals[index] = 0;
		//old_root->children[last_action] = nullptr;
		new_node->parent = nullptr;

		this->root.reset(new_node);
		//cout << "111" << endl;
	}
	else {
		this->root.reset(new TreeNode(nullptr, 1.));
	}
}

void MCTS::tree_deleter(TreeNode* t)
{
	//clock_t s = clock();
	if (t == nullptr)return;
	std::stack<TreeNode*> st;
	st.push(t);
	// remove children
	while (!st.empty())
	{
		TreeNode* top = st.top(); st.pop();
		for (int i = 0; i < top->legal_size; ++i)
			if (top->children[i])st.push(top->children[i]);
		// remove self
		delete[] top->children;
		delete[] top->legals;
		delete top;
	}
	//cout << (clock() - s) / (double)CLOCKS_PER_SEC << "s"<<endl;
}

std::vector<double> MCTS::get_action_probs(Board* board, double temp) {
	// submit simulate tasks to thread_pool
	std::vector<std::future<void>> futures;
	if (root->is_leaf)simulate(std::make_shared<Board>(*board));
	for (unsigned int i = 0; i < this->num_mcts_sims; i++) {
		// copy gomoku
		auto game = std::make_shared<Board>(*board);
		simulate(game);
		//board->printBoard(cout, NULL);
		//auto future =
			//this->thread_pool->commit(std::bind(&MCTS::simulate, this, game, num_mcts_sims));

		//future can't copy
		//futures.emplace_back(std::move(future));
	}
	// wait simulate
	for (unsigned int i = 0; i < futures.size(); i++) {
		//futures[i].wait();
	}

	// calculate probs
	std::vector<double> action_probs(board->get_action_size(), 0);
	const auto& children = this->root->children;
	if (!num_mcts_sims)
	{
		double sum = 0;
		for (int i = 0; i < root->legal_size; ++i)
			sum += action_probs[root->legals[i]] = children[i]->p_sa;
		for (double& p : action_probs)p /= sum;
		return action_probs;
	}
	//cout << (temp - 1e-3 < DBL_EPSILON) << endl;
	// greedy
	if (temp - 1e-3 < DBL_EPSILON) {
		unsigned int max_count = 0;
		unsigned int best_action = 0;
		for (int i = 0; i < root->legal_size; ++i)
		{
			if (children[i]->n_visited.load() > max_count) {
				max_count = children[i]->n_visited.load();
				best_action = root->legals[i];
			}
		}

		//for (unsigned int i = 0; i < children.size(); i++) {
		//	if (children[i] && children[i]->n_visited.load() > max_count) {
		//		max_count = children[i]->n_visited.load();
		//		best_action = i;
		//	}
		//}

		action_probs[best_action] = 1.;
		return action_probs;

	}
	else {
		// explore
		double sum = 0, sum1 = 0;
		//for (const auto& [i, child] : children) 
		for (int i = 0; i < root->legal_size; ++i) {
			if (children[i]->n_visited.load() > 0) {
				sum1 += children[i]->n_visited.load();
				//if (child->n_visited.load())
				//{
				//	cout << i << ' ' << child->n_visited.load() << ' '<<root->n_visited.load() << endl;
				//}
				//sum += action_probs[i];,km,l 
			}
		}
		//for (const auto& [i, child] : children)
		for (int i = 0; i < root->legal_size; ++i)
			if (children[i]->n_visited.load() > 0)
				sum += action_probs[root->legals[i]] = pow(children[i]->n_visited.load() / sum1, 1 / temp);
		//for (const auto& [i, child] : children)
		for (int i = 0; i < root->legal_size; ++i)
			if (children[i]->n_visited.load() > 0)
				action_probs[root->legals[i]] /= sum;
		//for (unsigned int i = 0; i < children.size(); i++) {
		//	if (children[i] && children[i]->n_visited.load() > 0) {
		//		action_probs[i] = (double)children[i]->n_visited.load() / (root->n_visited.load() - 1);
		//		//if (children[i] && children[i]->n_visited.load())
		//		//{
		//		//	cout << i << ' ' << children[i]->n_visited.load() << endl;
		//		//}
		//		//sum += action_probs[i];
		//	}
		//}
		//std::for_each(action_probs.begin(), action_probs.end(),
		//	[&sum, temp](double& x) { sum += x = pow(x, 1 / temp); });
		//// renormalization
		//std::for_each(action_probs.begin(), action_probs.end(),
		//	[sum](double& x) { x /= sum; });
		//for (unsigned int i = 0; i < children.size(); i++) {
		//	if (children[i] && children[i]->n_visited.load() > 0) {
		//		action_probs[i] = pow(children[i]->n_visited.load(), 1 / temp);
		//		sum += action_probs[i];
		//	}
			//if (children[i] && children[i]->n_visited.load())
			//{
			//	cout << i << ' ' << children[i]->n_visited.load() << endl;
			//}
	}
	//std::for_each(action_probs.begin(), action_probs.end(),
	//	[sum](double& x) { x /= sum; });

	return action_probs;
}
void MCTS::get_action_visits(Board* board, std::vector<unsigned>& action, std::vector<unsigned>& action_pruned)
{
	std::vector<std::future<void>> futures;
	if (root->is_leaf)simulate(std::make_shared<Board>(*board));
	for (unsigned int i = 0; i < this->num_mcts_sims; i++) {
		// copy gomoku
		auto game = std::make_shared<Board>(*board);
		//simulate(game);
		//board->printBoard(cout, NULL);
		auto future =
			this->thread_pool->commit(std::bind(&MCTS::simulate, this, game));

		//future can't copy
		futures.emplace_back(std::move(future));
	}
	// wait simulate
	for (unsigned int i = 0; i < futures.size(); i++) {
		futures[i].wait();
	}
	// calculate probs
	action.resize(board->get_action_size(), 0);
	action_pruned.resize(board->get_action_size(), 0);
	if (!root->legal_size)return;
	auto& children = this->root->children;
	unsigned mx = 0, mxi = 0;
	bool must = false;
	for (int i = 0; i < root->legal_size; ++i)
	{
		auto& child = children[i];
		if (!root->legals[i])continue;
		if (child->get_visited() && abs(child->q_sa - 1) < 1e-5) { must = true; break; }
		action[root->legals[i]] = child->n_visited;
		if (child->n_visited.load() > mx)
			mx = child->n_visited.load(), mxi = i;
	}
	if (must)
	{
		for (int i = 0; i < root->legal_size; ++i)
		{
			auto& child = children[i];
			if (!root->legals[i])continue;
			if (child->get_visited() && abs(child->q_sa - 1) < 1e-5)
				action[root->legals[i]] = action_pruned[root->legals[i]] = 1;
			else action[root->legals[i]] = action_pruned[root->legals[i]] = 0;
		}
		return;
	}
	//pruning
	if (forced)
		for (int i = 0; i < root->legal_size; ++i)
		{
			auto& child = children[i];
			if (!root->legals[i])continue;
			if (child->n_visited.load() > 0 && child->n_visited.load() < mx)
			{
				int n_force = child->get_n_force(this->root->n_visited);
				//cout << i<<' '<<n_force << endl;
				while (n_force && child->get_value(c_puct, 0, root->n_visited.load()) < children[mxi]->get_value(c_puct, 0, root->n_visited.load()))
				{
					if (!child->n_visited)break;
					--child->n_visited;
					--root->n_visited;
					--n_force;
				}
				if (child->n_visited.load() == 1)child->n_visited = 0, --root->n_visited;
			}
			action_pruned[root->legals[i]] = child->n_visited;
		}
	else action_pruned = action;
}

void MCTS::simulate(std::shared_ptr<Board> game) {
	// execute one simulation
	auto node = this->root.get();
	while (true) {
		if (node->get_is_leaf()) {
			break;
		}
		unsigned index = 0;
		if (forced && node == root.get())
			index = node->select_forced(this->c_puct, this->c_virtual_loss);
		// select
		else index = node->select(this->c_puct, this->c_virtual_loss);
		game->playPolicy(node->legals[index]);
		node = node->children[index];
	}
	// get game status
	auto winner = game->winner;
	double value = 0;
	// not end
	if (winner == C_WALL) {
		// predict action_probs and value by neural network

		//auto future = this->neural_network->forward();
		//auto result = future.get();
		//std::vector <double> action_priors(Board::get_action_size(), 1. / Board::get_action_size());
		//std::default_random_engine e(game->raw_pos_hash.hash0 ^ time(nullptr) ^ clock());
		//std::uniform_real_distribution u(-1., 1.);
		//value = u(e);
		//cout << value << endl;
		static std::vector<float> action_priors = std::vector<float>(Board::get_action_size(), 1. / Board::get_action_size());
		if (!neural_network->model)
		{
			std::random_device e;
			std::uniform_real_distribution u(-1., 1.);
			//action_priors = std::vector<float>(Board::get_action_size(), 1. / Board::get_action_size());
			value = u(e);
		}
		else
		{
			auto tmp = boardToState(*game, false);
			std::future<NetWork::result_type> future = neural_network->pushState(tmp.first, tmp.second);
			auto [_priors, _val] = future.get();
			action_priors = _priors;
			value = _val;
		}
		//cout << value << endl;;
		//{
		//    std::lock_guard<std::mutex> lock(this->lock);

		//    result = neural_network->forward(torch::jit::Stack({ tmp.first.unsqueeze(0),tmp.second.unsqueeze(0) })).toTuple()->elements();
		//}
		//auto action = result[0].toTensor().view(-1).to(at::kCPU);
		//cout << *action.data_ptr<float>() << endl;
		//std::vector<double> action_priors(static_cast<float*>(action.data_ptr()), static_cast<float*>(action.data_ptr()) + action.numel());

		if (noise && root->get_is_leaf())
		{
			std::vector<float> dir = dirichlet(action_priors.size(), alpha);
			double sum = 0;
			//cout << *std::max_element(action_priors.begin(), action_priors.end()) << endl;
			for (int i = 0; i < action_priors.size(); ++i)
			{
				//auto ori = action_priors[i];
				action_priors[i] = .75 * action_priors[i] + .25 * dir[i];
				//if (ori > DBL_EPSILON)
				//	cout << ori << ' ' << action_priors[i] << endl;
			}
			//std::sort(action_priors.begin(), action_priors.end(), std::greater<double>());
			//cout << action_priors[0] << ' ' << action_priors[1] << endl;

		}
		//value = *result[1].toTensor().view(-1).to(at::kCPU).data_ptr<float>();
		// mask invalid actions
		//auto legal_moves = game->get_legal_moves();
		auto legal_pols = game->genLegalPolicy();
		std::vector<float> legal_probs;
		double sum = 0;
		for (const auto& pol : legal_pols)
			sum += action_priors[pol];
		if (legal_pols.empty())value = -1;
		else
		{
			//for (unsigned int i = 0; i < action_priors.size(); i++) {
			//	if (legal_moves[i] == 1) {
			//		sum += action_priors[i];
			//	}
			//	else {
			//		action_priors[i] = 0;
			//	}
			//}
			// renormalization
			if (sum > DBL_EPSILON) {
				for (const auto& pol : legal_pols)
					legal_probs.push_back(action_priors[pol] / sum);
				//std::for_each(action_priors.begin(), action_priors.end(),
				//	[sum](double& x) { x /= sum; });
			}
			else {
				// all masked

				// NB! All valid moves may be masked if either your NNet architecture is
				// insufficient or you've get overfitting or something else. If you have
				// got dozens or hundreds of these messages you should pay attention to
				// your NNet and/or training process.
				std::cout << "All valid moves were masked, do workaround." << std::endl;

				cout << sum << endl;
				if (game->nextPla == P_BLACK)
					for (const auto& pol : legal_pols)
						cout << to_string(Location::policy[pol]) << ' ' << action_priors[pol] << endl;
				else for (const auto& pol : legal_pols)
					cout << to_string(Location::policy[pol].rot_()) << ' ' << action_priors[pol] << endl;
				cout << *game << endl;

				for (const auto& pol : legal_pols)
					legal_probs.push_back(1. / legal_pols.size());
				//sum = std::accumulate(legal_moves.begin(), legal_moves.end(), 0);
				//for (unsigned int i = 0; i < action_priors.size(); i++) {
				//	action_priors[i] = legal_moves[i] / sum;
				//}
			}

			// expand
			node->expand(legal_pols, legal_probs);
		}
	}
	else {
		// end
		value = (winner == C_EMPTY ? 0 : (winner == game->nextPla ? 1 : -1));
		//cout << game->nextPla<<' '<<winner << endl;
	}
	// value(parent -> node) = -value
	node->backup(-value);
	return;
}
