#pragma once
#include <torch/torch.h>

class MyModel : public torch::nn::Module {
 public:
  MyModel();
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Conv2d conv1 = nullptr;
  torch::nn::Conv2d conv2 = nullptr;
  torch::nn::Dropout2d conv2_drop;
  torch::nn::Linear fc1 = nullptr;
  torch::nn::Linear fc2 = nullptr;
};