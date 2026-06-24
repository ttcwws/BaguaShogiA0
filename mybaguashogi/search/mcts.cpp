#include <cmath>
#include <cfloat>
#include <numeric>
#include <iostream>
#include <stack>
#include <chrono>

#include "mcts.h"
#include "../core/stable_time.h"

using namespace std::chrono_literals;
using std::move;

NetWorkParallelizer::~NetWorkParallelizer()
{
	running = false;
}
bool NetWorkParallelizer::try_get_cache(const Hash128& hash, result_type& result)
{
	const size_t index = hash.hash0 & NN_CACHE_MASK;
	const size_t mutex_index = index & NN_CACHE_MUTEX_POOL_MASK;
	std::shared_ptr<const result_type> cached;
	{
		std::lock_guard<std::mutex> guard(nn_cache_mutex_pool[mutex_index]);
		const CacheEntry& entry = nn_cache[index];
		if (!entry.value || entry.hash != hash)
			return false;
		cached = entry.value;
	}
	result = *cached;
	nn_cache_hits.fetch_add(1, std::memory_order_relaxed);
	return true;
}
void NetWorkParallelizer::put_cache(const Hash128& hash, const result_type& result)
{
	std::shared_ptr<const result_type> cached = std::make_shared<result_type>(result);
	const size_t index = hash.hash0 & NN_CACHE_MASK;
	const size_t mutex_index = index & NN_CACHE_MUTEX_POOL_MASK;
	{
		std::lock_guard<std::mutex> guard(nn_cache_mutex_pool[mutex_index]);
		nn_cache[index].hash = hash;
		nn_cache[index].value.swap(cached);
	}
}
NetWorkParallelizer::result_type NetWorkParallelizer::predict(const Board& board)
{
	if (!model)
	{
		size_t s = board.get_action_size();
		return std::make_pair(std::vector<float>(s, 1.f / s), 0.f);
	}

	const Hash128 nn_hash = board.raw_pos_hash;
	result_type cached_result;
	if (try_get_cache(nn_hash, cached_result))
		return cached_result;
	nn_cache_misses.fetch_add(1, std::memory_order_relaxed);

	std::promise<result_type> promise;
	std::vector<int8_t> bin;
	std::vector<float> overall;
	std::future<result_type> fut = promise.get_future();
	boardToStateRaw(board, bin, overall);
	tasks.enqueue({ move(promise), move(bin), move(overall) });
	result_type result = fut.get();
	put_cache(nn_hash, result);
	return result;
}
void NetWorkParallelizer::get_data()
{
	//if (inferable)return;
	static size_t b = 0;
	static auto t = stable_time();
	static size_t count = 0;
	std::vector<Buffer::inputType> t_bin, t_overall;
	std::vector<std::promise<result_type>> proms;
	std::vector<std::tuple<std::promise<result_type>, std::vector<int8_t>, std::vector<float>>> task;
	task.reserve(MAX_BATCH_SIZE);
	Buffer* buf = nullptr;
	std::unique_lock<std::mutex> lk(lock);
	while (running && proms.size() < MAX_BATCH_SIZE)
	{
		int s = 0;
		task.resize(MAX_BATCH_SIZE - proms.size());
		if (s = tasks.try_dequeue_bulk(task.begin(), task.size()))
		{
			if (s + proms.size() == MAX_BATCH_SIZE)lk.unlock();
			task.resize(s);
			//cout << s << endl;
			size_t oldsize = proms.size();
			proms.resize(proms.size() + s);
			t_bin.resize(t_bin.size() + BINARY_SIZE * s);
			t_overall.resize(t_overall.size() + OVERALL_SIZE * s);
			std::for_each_n(std::execution::par_unseq, IndexIterator(), s, [&](const size_t& i) {
				auto& [prom, x, y] = task[i];
				proms[oldsize + i].swap(prom);
				std::transform(
					std::execution::par_unseq,
					x.begin(),
					x.end(),
					t_bin.begin() + (oldsize + i) * BINARY_SIZE,
					[](int8_t v) { return Buffer::inputType(v); }
				);
				std::transform(
					std::execution::par_unseq,
					y.begin(),
					y.end(),
					t_overall.begin() + (oldsize + i) * OVERALL_SIZE,
					[](float v) { return Buffer::inputType(v); }
				);
				});
		}
		else if (!proms.empty() && (buf = bufpool.get_buffer())) break;
		else continue;
	}
	if (lk.owns_lock())lk.unlock();
	if (!running)return;
	thp.commit(std::bind(&NetWorkParallelizer::get_data, this));
	while (!buf)
	{
		std::this_thread::yield();
		buf = bufpool.get_buffer();
	}
	//cout_mutex.lock();
	//cout << Buffer::waiting_num << endl;
	//cout_mutex.unlock();
	auto [logits, values] = buf->infer(t_bin, t_overall);
	{
		std::lock_guard<std::mutex> lk(cout_mutex);
		b += proms.size();
		++count;
		if (b > 300000)
		{
			auto now = stable_time();
			double s = to_seconds_double(now - t);
			std::cout << "Inference speed: " << double(b) / s << " samples/s, Avg Batch Size: " << double(b) / count << std::endl;
			std::cout << "Buffer pool size:  " << bufpool.size() << std::endl;
			const uint64_t hits = nn_cache_hits.load(std::memory_order_relaxed);
			const uint64_t misses = nn_cache_misses.load(std::memory_order_relaxed);
			const uint64_t lookups = hits + misses;
			if (lookups)
				std::cout << "NN cache hit rate: " << (100.0 * hits / lookups) << "% (" << hits << "/" << lookups << ")" << std::endl;
			b = count = 0;
			t = now;
		}
	}
	std::for_each_n(std::execution::par_unseq, IndexIterator(), proms.size(),
		[&logits, &values, &proms, this](const size_t& i)
		{
			std::vector<float> probs;
			softmax(logits.begin() + i * POLICY_SIZE, probs, temperature);
			proms[i].set_value(std::make_pair(
				std::move(probs),
				static_cast<float>(values[i])
			));
		});

}
// TreeNode
TreeNode::TreeNode()
	: parent(nullptr),
	is_leaf(true),
	n_visited(0),
	p_sa(0),
	p_ex(0),
	value_sum(0),
	determine(0),
	value_self(0),
	action(-1),
	children() {
}

TreeNode::TreeNode(TreeNode* parent, unsigned action, float p_sa)
	: parent(parent),
	is_leaf(true),
	n_visited(0),
	value_sum(0),
	p_sa(p_sa),
	p_ex(0),
	determine(0),
	value_self(0),
	action(action),
	children() {
}

TreeNode::TreeNode(const TreeNode& node)
{
	this->parent = node.parent;
	this->children = node.children;
	this->is_leaf = node.is_leaf;
	this->n_visited = node.n_visited;
	this->p_sa = node.p_sa;
	this->p_ex = node.p_ex;
	this->value_sum = node.value_sum;
	this->determine = node.determine;
	this->value_self = node.value_self;
	this->action = node.action;
}

TreeNode& TreeNode::operator=(const TreeNode& node) {
	if (this == &node) return *this;
	this->parent = node.parent;
	this->children = node.children;
	this->is_leaf = node.is_leaf;
	this->n_visited = node.n_visited;
	this->p_sa = node.p_sa;
	this->p_ex = node.p_ex;
	this->value_sum = node.value_sum;
	this->determine = node.determine;
	this->value_self = node.value_self;
	this->action = node.action;
	return *this;
}

TreeNode* TreeNode::select(float c_puct, float c_fpu) {
	float best_value = -FLT_MAX;
	TreeNode* best_node = nullptr;
	if (!children.empty())
	{
		for (const auto& child : children)
		{
			if (!child || child->determine > 0)continue;
			const float cur_value = c_puct * child->p_sa * sqrtf(n_visited) / (1 + child->n_visited) - child->get_value();
			if (cur_value > best_value) {
				best_value = cur_value;
				best_node = child.get();
			}
		}
	}
	//for (const auto& child : children)
	//{
	//	if (!child)continue;
	//	float u = c_puct * child->p_sa * sqrt(n_visited) / (1 + child->n_visited);
	//	float cur_value = u - child->get_value();
	//	if (cur_value > best_value) {
	//		best_value = cur_value;
	//		best_node = child.get();
	//	}
	//}
	if (!lazy_children.empty())
	{
		const auto& [action, p_sa] = lazy_children.back();
		float lazy_value = c_puct * p_sa * sqrtf(n_visited) + get_value();
		if (lazy_value > best_value)
		{
			p_ex += p_sa;
			children.push_back(std::make_shared<TreeNode>(this, action, p_sa));
			lazy_children.pop_back();
			return children.back().get();
		}
	}
	if (!best_node)
	{
		cout << children.size() + lazy_children.size() << endl;
		cout << "Nullptr select" << endl;
		cout << "children:" << endl;
		for (const auto& child : children)
		{
			// empty node
			if (!child)continue;
			float u = c_puct * child->p_sa * sqrt(n_visited) / (1 + child->n_visited);
			float cur_value = u - child->get_value();
			cout << child->p_sa << ' ' << child->get_value() << endl;
		}
		cout << "lazy children:" << endl;
		for (const auto& child : lazy_children)
		{
			// empty node
			cout << child.second << endl;
		}
	}

	return best_node;
}
unsigned TreeNode::get_n_force(unsigned sum_n_visited)
{
	return (unsigned)sqrt(1.8 * p_sa * sum_n_visited);
}
unsigned TreeNode::get_visited() const
{
	return n_visited;
}
TreeNode* TreeNode::select_forced(float c_puct) {
	//throw '?';
	double best_value = -DBL_MAX;
	TreeNode* best_node = nullptr;
	for (const auto& child : children)
	{
		// empty node
		unsigned int sum_n_visited = n_visited;
		if (child->n_visited < child->get_n_force(sum_n_visited))
		{
			best_node = child.get();
			break;
		}
		//cout << child->p_sa << ' ' << sum_n_visited << ' ' << child->get_n_force(sum_n_visited) << endl;
		double u = c_puct * child->p_sa * sqrt(n_visited) / (1 + child->n_visited);
		double cur_value = u - child->get_value();
		if (cur_value > best_value) {
			//if (best_node)--best_node->virtual_loss;
			best_value = cur_value;
			best_node = child.get();
			//++best_node->virtual_loss;
			// add vitural loss
		}
	}
	if (!best_node)
	{
		cout << children.size() << endl;
		cout << "Nullptr select force" << endl;
		for (const auto& child : children)
		{
			// empty node
			if (!child)continue;
			double u = c_puct * child->p_sa * sqrt(n_visited) / (1 + child->n_visited);
			double cur_value = u - child->get_value();
			cout << child->p_sa << ' ' << child->get_value() << endl;
		}
	}
	return best_node;
}

void TreeNode::expand(const std::vector<size_t>& legal_pols, const std::vector<float>& policy_probs)
{
	if (is_leaf) {
		int n = legal_pols.size();
		children.clear();
		for (int i = 0; i < n; ++i) {
			lazy_children.push_back(std::make_pair(legal_pols[i], policy_probs[i]));
			if (legal_pols[i] == -1 || isnan(policy_probs[i]))
				std::cout << "illegal action in expand: " << legal_pols[i] << std::endl;
		}
		is_leaf = false;
		if (lazy_children.empty())
		{
			cout << "expand with no legal moves" << endl;
		}
	}
}
void TreeNode::sort_children()
{
	if (is_leaf)return;
	std::sort(lazy_children.begin(), lazy_children.end(), [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {return a.second < b.second; });
}
void TreeNode::add_dirichlet_noise(float alpha, float epsilon, float t)
{
	if (is_leaf)return;
	const size_t total_children = children.size() + lazy_children.size();
	std::vector<float> dir = dirichlet(total_children, alpha);
	float sum = 0, k;
	const float inv = .1f / sqrt(total_children);
	for (int i = 0; i < children.size(); ++i)
		sum += children[i]->p_sa = max(pow(children[i]->p_sa, 1.f / t), inv);
	for (int i = 0; i < lazy_children.size(); ++i)
		sum += lazy_children[i].second = max(pow(lazy_children[i].second, 1.f / t), inv);
	k = (1 - epsilon) / sum;
	for (int i = 0; i < children.size(); ++i)
		children[i]->p_sa = children[i]->p_sa * k + dir[i] * epsilon;
	for (int i = 0; i < lazy_children.size(); ++i)
		lazy_children[i].second = lazy_children[i].second * k + dir[i + children.size()] * epsilon;
}
void TreeNode::trim()
{
	if (is_leaf)return;
	bool winning = false, losing = lazy_children.empty();
	int min_win = INT_MAX, max_lose = INT_MIN;
	for (const auto& child : children)
		child->trim();
	for (const auto& child : children)
	{
		//child->trim();
		if (child->determine)
			if (child->determine < 0)
				winning = true, min_win = std::min(min_win, -child->determine);
			else if (child->determine > 0)
				max_lose = std::max(max_lose, child->determine);
		losing = losing && child->determine > 0;
	}
	std::vector<std::shared_ptr<TreeNode>> new_children;
	new_children.reserve(children.size());
	if (winning)
	{
		for (const auto& child : children)
			if (child->determine < 0 && -child->determine == min_win)
			{
				child->n_visited = 1;
				child->value_sum = -1;
				new_children.push_back(child);
			}
		lazy_children.clear();
		determine = min_win + 1;
	}
	else if (losing)
	{
		for (const auto& child : children)
			if (child->determine > 0 && child->determine == max_lose)
			{
				child->n_visited = 1;
				child->value_sum = 1;
				new_children.push_back(child);
			}
		determine = -max_lose - 1;
	}
	else
	{
		for (const auto& child : children)
			if (child->determine <= 0)
				new_children.push_back(child);
	}
	if (determine)
		value_self = determine > 0 ? 1 : -1;
	n_visited = std::accumulate(new_children.begin(), new_children.end(), 0U, [](unsigned sum, const std::shared_ptr<TreeNode>& child) {return sum + child->n_visited; }) + 1;
	value_sum = -std::accumulate(new_children.begin(), new_children.end(), 0.f, [](float sum, const std::shared_ptr<TreeNode>& child) {return sum + child->value_sum; }) + value_self;
	children.swap(new_children);
}

void TreeNode::backup(float value) {
	// If it is not root, this node's parent should be updated first
	TreeNode* node = this;
	value_self = value;
	while (node) {
		node->n_visited++;
		node->value_sum += value;
		value = -value;
		node = node->parent;
	}
}

float TreeNode::get_value() const {
	return n_visited ? value_sum / n_visited : 0.f;
}

// MCTS
MCTS::MCTS(NetWork* neural_network, float c_puct,
	unsigned int num_mcts_sims,
	unsigned int action_size, float alpha, bool forced, bool gpu, bool noise)
	: neural_network(neural_network),
	c_puct(c_puct),
	num_mcts_sims(num_mcts_sims),
	action_size(action_size),
	alpha(alpha),
	use_gpu(gpu),
	forced(forced),
	noise(noise),
	root(new TreeNode(nullptr, -1, 1)) {
}
void MCTS::update_with_move(int last_action) {
	auto old_root = this->root.get();
	// reuse the child tree
	auto child = std::find_if(old_root->children.begin(), old_root->children.end(),
		[last_action](const std::shared_ptr<TreeNode>& node) { return last_action == node->action; });
	if (child != old_root->children.end()) {
		// unlink
		this->root.swap(*child);
		this->root->parent = nullptr;
		child->reset();

		if (noise && !root->is_leaf)
			root->add_dirichlet_noise(alpha), root->sort_children();
	}
	else this->root.reset(new TreeNode(nullptr, -1, 1.));
}

void MCTS::get_action_visits(const Board& board, std::vector<unsigned>& action, std::vector<unsigned>& action_pruned)
{
	if (root->is_leaf)simulate(board);
	if (!root->determine)
	{
		for (unsigned int i = 0; i < this->num_mcts_sims; i++) {
			simulate(board);
		}
		if (root->children.empty() && root->lazy_children.empty())
		{
			cout << "no legal moves before trim" << endl;
			return;
		}
		root->trim();
	}
	// wait simulate
	// calculate probs
	const size_t action_size = board.get_action_size();
	if (action.size() != action_size)
		action.assign(action_size, 0);
	else
		std::fill(action.begin(), action.end(), 0);
	if (action_pruned.size() != action_size)
		action_pruned.assign(action_size, 0);
	else
		std::fill(action_pruned.begin(), action_pruned.end(), 0);
	if (root->children.empty())
	{
		if (root->lazy_children.empty())
		{
			cout << board << endl;
			cout << root->determine << endl;
			return;
		}
		for (const auto& [a, p] : root->lazy_children)
			action[a] = 1;
		action_pruned = action;
		return;
	}
	auto& children = root->children;
	for (const auto& child : children)
		action[child->action] = child->n_visited;
	//std::for_each(children.begin(), children.end(),
		//[&action](const std::shared_ptr<TreeNode>& child) { action[child->action] = child->n_visited; });
	//pruning
	//if (forced)
	//	for (const auto& child : children)
	//	{
	//		if (child->n_visited > 0 && child->n_visited < mx)
	//		{
	//			int n_force = child->get_n_force(this->root->n_visited);
	//			//cout << i<<' '<<n_force << endl;
	//			while (n_force &&
	//				c_puct * child->p_sa * sqrt(root->n_visited) / (1 + child->n_visited) - child->get_value() <
	//				c_puct * mxc->p_sa * sqrt(root->n_visited) / (1 + mxc->n_visited) - mxc->get_value())
	//			{
	//				if (!child->n_visited)break;
	//				--child->n_visited;
	//				--root->n_visited;
	//				--n_force;
	//			}
	//			if (child->n_visited == 1)child->n_visited = 0, --root->n_visited;
	//		}
	//		action_pruned[child->action] = child->n_visited;
	//	}
	action_pruned = action;
}

void MCTS::simulate(Board game) {
	// execute one simulation
	auto node = root.get();
	while (!node->get_is_leaf()) {
		TreeNode* child = nullptr;
		float c = this->c_puct;
		// c *= sqrt((node->children.size() + node->lazy_children.size()) / 100.f);
		child = node->select(c, noise && node == root.get() ? 0.f : .2f);
		game.playPolicy(child->action);
		node = child;
	}
	// get game status
	auto winner = game.winner;
	float value = 0;
	// not end
	if (winner == C_WALL) {
		// predict action_probs and value by neural network

		auto legal_pols = game.genLegalPolicy();
		if (legal_pols.empty())node->determine = value = -1;
		else if (game.movenum >= 500 && !game.attacked)value = 0;
		else
		{
			auto [action_priors, _val] = neural_network->predict(game);
			value = _val;
			std::vector<float> legal_probs;
			legal_probs.reserve(legal_pols.size());
			float sum = 0;
			for (const auto& pol : legal_pols)
				sum += action_priors[pol];
			// renormalization
			if (sum > FLT_EPSILON)
				for (const auto& pol : legal_pols)
					legal_probs.push_back(action_priors[pol] / sum);
			else {
				// all masked

				// NB! All valid moves may be masked if either your NNet architecture is
				// insufficient or you've get overfitting or something else. If you have
				// got dozens or hundreds of these messages you should pay attention to
				// your NNet and/or training process.
				{
					std::lock_guard<std::mutex> _(cout_mutex);
					std::cout << "All valid moves were masked, do workaround." << std::endl;
					//for (const auto& pol : legal_pols)
					//	cout << action_priors[pol] << endl;
					cout << sum << endl;
					if (game.nextPla == P_BLACK)
						for (const auto& pol : legal_pols)
							cout << to_string(Location::policy[pol]) << ' ' << action_priors[pol] << endl;
					else for (const auto& pol : legal_pols)
						cout << to_string(Location::policy[pol].rot_()) << ' ' << action_priors[pol] << endl;
					cout << game << endl;
					for (int p = 0; p < action_priors.size(); ++p)
						if (action_priors[p] > .01)
							cout << to_string(Location::policy[p]) << ' ' << action_priors[p] << endl;
					cout << endl;

					for (const auto& pol : legal_pols)
						legal_probs.push_back(1. / legal_pols.size());
				}
			}

			// expand
			node->expand(legal_pols, legal_probs);
			if (noise && node == root.get())
				node->add_dirichlet_noise(this->alpha);
			node->sort_children();
		}
	}
	else {
		// end
		node->determine = value = (winner == C_EMPTY ? 0 : (winner == game.nextPla ? 1 : -1));
	}
	node->backup(value);
	return;
}
