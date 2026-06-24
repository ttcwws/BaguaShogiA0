//#pragma once
//#include <torch/scripts.h>
//constexpr auto INPUT_AREA = 81;
//constexpr auto POLICY_SIZE = 4616;
//
//using namespace torch;
//namespace F = torch::nn::functional;
//torch::nn::Conv2dOptions conv_options(int in_planes, int out_planes, int kernel_size,
//    int padding = 0, bool with_bias = false);
//class ResBlock :nn::Module
//{
//public:
//    ResBlock(int num_hidden);
//    Tensor forward(const Tensor& x);
//private:
//    nn::Conv2d conv1, conv2;
//    nn::BatchNorm2d bn1, bn2;
//};
//class ResNet :nn::Module
//{
//public:
//    ResNet(int input_channels_bin, int input_channels_overall, int num_resBlocks, int num_hidden);
//    Tensor forward(const Tensor& binary, const Tensor& overall);
//private:
//    nn::Conv2d startBin;
//    nn::Linear startOverall;
//    nn::ModuleList backBone;
//    nn::Sequential startBlock, policyHead, valueHead;
//};