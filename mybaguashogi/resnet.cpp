//#include "resnet.h"
//using nn::Conv2dOptions;
//using nn::BatchNorm2dOptions;
//Conv2dOptions conv_options(int in_planes, int out_planes, int kernel_size,
//    int padding = 0, bool with_bias = false)
//{
//    return Conv2dOptions(in_planes, out_planes, kernel_size).bias(with_bias).padding(padding);
//}
//ResBlock::ResBlock(int num_hidden) :
//    conv1(conv_options(num_hidden, num_hidden, 3, 1)),
//    bn1(BatchNorm2dOptions(num_hidden)),
//    conv2(conv_options(num_hidden, num_hidden, 3, 1)),
//    bn2(BatchNorm2dOptions(num_hidden)) {}
//
//Tensor ResBlock::forward(const Tensor& x)
//{
//    Tensor tmp = x;
//    tmp = F::mish(bn1(conv1(x)));
//    return F::mish(bn2(conv2(tmp)) + x);
//}
//
//ResNet::ResNet(int input_channels_bin, int input_channels_overall, int num_resBlocks, int num_hidden) :
//    startBin(conv_options(input_channels_bin, num_hidden, 3, 1)),
//    startOverall(input_channels_overall, num_hidden),
//    startBlock(nn::BatchNorm2d(BatchNorm2dOptions(num_hidden)),
//        nn::Mish()),
//    backBone(),
//    policyHead(nn::Conv2d(conv_options(num_hidden, 32, 3, 1)),
//        nn::BatchNorm2d(BatchNorm2dOptions(32)),
//        nn::Mish(),
//        nn::Flatten(),
//        nn::Linear(32 * INPUT_AREA, POLICY_SIZE),
//        nn::LogSoftmax()),
//    valueHead(nn::Conv2d(conv_options(num_hidden, 8, 3, 1)),
//        nn::BatchNorm2d(BatchNorm2dOptions(8)),
//        nn::Mish(),
//        nn::Flatten(),
//        nn::Linear(8 * INPUT_AREA, 64),
//        nn::Mish(),
//        nn::Linear(64, 1),
//        nn::Tanh())
//{
//    for (int i = 0; i < num_resBlocks; ++i)
//        backBone->push_back(ResBlock(num_hidden));
//}
//
//Tensor ResNet::forward(const Tensor& binary, const Tensor& overall)
//{
//    Tensor x_bin = startBin(binary),
//        x_overall = startOverall(overall).unsqueeze(-1).unsqueeze(-1);
//    Tensor x = startBlock->forward(x_bin + x_overall);
//    for (auto& resBlock : *backBone)
//    {
//    }
//}
//
//    def forward(self, binary, overall) :
//    x_bin = self.startBin(binary)
//    x_overall = self.startOverall(overall).unsqueeze(-1).unsqueeze(-1)
//    x = x_bin + x_overall
//    x = self.startBlock(x)
//    for resBlock in self.backBone :
//        x = resBlock(x)
//        policy = self.policyHead(x)
//        value = self.valueHead(x)
//        return policy, value
