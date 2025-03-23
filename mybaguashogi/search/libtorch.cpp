#pragma once
#include "libtorch.h"
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAGuard.h>

#include <iostream>

using namespace std::chrono_literals;
std::pair<Tensor, Tensor> boardToState(const Board& board, bool use_gpu)
{
	constexpr size_t bin_num = 57, overall_num = 15;
	float bin[bin_num][9][9]{ 0 }, overall[overall_num]{ 0 };
	auto dv = use_gpu ? at::kCUDA : at::kCPU;
	//Tensor bin = at::zeros({ bin_num, board.x_size, board.y_size }, opt), overall = at::zeros({ 16 }, opt);
	//0 : on board mask
	//1~27: friendly pieces
	//28~54: enemy pieces
	//55, 56: self king and opponent king location

	//**** new state ****
	//0 : on board mask
	//1~14: friendly pieces
	//15~28: enemy pieces
	//29,30: self king and opponent king location
	Player pla = board.nextPla;
	Player opp = getOpp(pla);
	for (int i = 0; i < board.x_size; ++i)
		for (int j = 0; j < board.y_size; ++j)
		{
			int x = i, y = j;
			if (pla == P_WHITE)x = board.x_size - 1 - i, y = board.y_size - 1 - j;
			Loc loc = Location::getLoc(x, y, board.x_size);
			Piece pc = board.colors[loc];
			Piece tp = getType(pc);
			if (board.isOnBoard(i, j))
				bin[0][i][j] = 1;
			if (pc != C_EMPTY && pc != C_WALL)
			{
				//float val = 1;
				//Piece ch = tp;
				//if (getPromoted(pc))
				//	ch += 6;
				//if (getHorizontal(pc))
				//	val = -1;
				//if (getSide(pc) == pla)
				//	bin[ch][i][j] = val;
				//else if (getSide(pc) == opp)
				//	bin[ch + 15][i][j] = val;
				Piece ch = tp;
				if (getPromoted(pc))
					ch += 6;
				if (getHorizontal(pc))
					ch += 13;
				if (getSide(pc) == pla)
					bin[ch][i][j] = 1;
				else if (getSide(pc) == opp)
					bin[ch + 27][i][j] = 1;
			}
			//if (loc == board.KING_INIT_LOCATION[pla])
			//	bin[29][i][j] = 1;

			//if (loc == board.KING_INIT_LOCATION[opp])
			//	bin[30][i][j] = 1;
			if (loc == board.KING_INIT_LOCATION[pla])
				bin[55][i][j] = 1;

			if (loc == board.KING_INIT_LOCATION[opp])
				bin[56][i][j] = 1;
		}
	//0, 1 : if self king and oppoenent king is attacked
	//2~8 : self koma
	//9~15 : opponent koma
	for (Piece i = S_KING + 1; i <= S_PAWN; ++i)
		if (i == S_PAWN)
		{
			overall[i - 1] = board.koma[pla][i] / 3.;
			overall[i + 6] = board.koma[opp][i] / 3.;
		}
		else
		{
			overall[i - 1] = board.koma[pla][i] / 2.;
			overall[i + 6] = board.koma[opp][i] / 2.;
		}
	if (GameLogic::locAttacked(board, board.kings[pla], opp))overall[0] = 1;
	//,cout << at::from_blob(bin, { 16 }, opt).clone() << endl;
	return std::make_pair(at::from_blob(bin, { bin_num,9,9 }, torch::kFloat32).to(dv).clone(),
		at::from_blob(overall, { overall_num }, torch::kFloat32).to(dv).clone());
}
//NeuralNetwork::NeuralNetwork(std::string model_path, bool use_gpu,
//                             unsigned int batch_size)
//    : module(std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path.c_str()))),
//      use_gpu(use_gpu),
//      batch_size(batch_size),
//      running(true),
//      loop(nullptr) {
//  if (this->use_gpu) {
//    // move to CUDA
//    this->module->to(at::kCUDA);
//  }
//
//  // run infer thread
//  this->loop = std::make_unique<std::thread>([this] {
//    while (this->running) {
//      this->infer();
//    }
//  });
//}
//
//NeuralNetwork::~NeuralNetwork() {
//  this->running = false;
//  this->loop->join();
//}
//
//std::future<NeuralNetwork::return_type> NeuralNetwork::commit(Board* game) {
//  //int n = game->get_n();
//    int n;
//
//  // convert data format
//  auto board = game->get_board();
//  std::vector<int> board0;
//  for (unsigned int i = 0; i < board.size(); i++) {
//    board0.insert(board0.end(), board[i].begin(), board[i].end());
//  }
//
//  torch::Tensor temp =
//      torch::from_blob(&board0[0], {1, 1, n, n}, torch::dtype(torch::kInt32));
//
//  torch::Tensor state0 = temp.gt(0).toType(torch::kFloat32);
//  torch::Tensor state1 = temp.lt(0).toType(torch::kFloat32);
//
//  int last_move = game->get_last_move();
//  int cur_player = game->get_current_color();
//
//  if (cur_player == -1) {
//    std::swap(state0, state1);
//  }
//
//  torch::Tensor state2 =
//      torch::zeros({1, 1, n, n}, torch::dtype(torch::kFloat32));
//
//  if (last_move != -1) {
//    state2[0][0][last_move / n][last_move % n] = 1;
//  }
//
//  // torch::Tensor states = torch::cat({state0, state1}, 1);
//  torch::Tensor states = torch::cat({state0, state1, state2}, 1);
//
//  // emplace task
//  std::promise<return_type> promise;
//  auto ret = promise.get_future();
//
//  {
//    std::lock_guard<std::mutex> lock(this->lock);
//    tasks.emplace(std::make_pair(states, std::move(promise)));
//  }
//
//  this->cv.notify_all();
//
//  return ret;
//}
//
//// TODO: use lock-free queue
//// https://github.com/cameron314/concurrentqueue
//void NeuralNetwork::infer() {
//  // get inputs
//  std::vector<torch::Tensor> states;
//  std::vector<std::promise<return_type>> promises;
//
//  bool timeout = false;
//  while (states.size() < this->batch_size && !timeout) {
//    // pop task
//    {
//      std::unique_lock<std::mutex> lock(this->lock);
//      if (this->cv.wait_for(lock, 1ms,
//                            [this] { return this->tasks.size() > 0; })) {
//        while (states.size() < this->batch_size && this->tasks.size() > 0){
//          auto task = std::move(this->tasks.front());
//          states.emplace_back(std::move(task.first));
//          promises.emplace_back(std::move(task.second));
//
//          this->tasks.pop();
//        }
//
//      } else {
//        // timeout
//        // std::cout << "timeout" << std::endl;
//        timeout = true;
//      }
//    }
//  }
//
//  // inputs empty
//  if (states.size() == 0) {
//    return;
//  }
//
//  // infer
//  std::vector<torch::jit::IValue> inputs{
//      this->use_gpu ? torch::cat(states, 0).to(at::kCUDA)
//                    : torch::cat(states, 0)};
//  auto result = this->module->forward(inputs).toTuple();
//
//  torch::Tensor p_batch = result->elements()[0]
//                              .toTensor()
//                              .exp()
//                              .toType(torch::kFloat32)
//                              .to(at::kCPU);
//  torch::Tensor v_batch =
//      result->elements()[1].toTensor().toType(torch::kFloat32).to(at::kCPU);
//
//  // set promise value
//  for (unsigned int i = 0; i < promises.size(); i++) {
//    torch::Tensor p = p_batch[i];
//    torch::Tensor v = v_batch[i];
//
//    std::vector<double> prob(static_cast<float*>(p.data_ptr()),
//                             static_cast<float*>(p.data_ptr()) + p.size(0));
//    std::vector<double> value{v.item<float>()};
//    return_type temp{std::move(prob), std::move(value)};
//
//    promises[i].set_value(std::move(temp));
//  }
//}
