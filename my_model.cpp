#include "my_model.h"

MyModel::MyModel() {
  conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5));
  conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 5));
  fc1 = torch::nn::Linear(320, 50);
  fc2 = torch::nn::Linear(50, 10);

  register_module("conv1", conv1);
  register_module("conv2", conv2);
  register_module("conv2_drop", conv2_drop);
  register_module("fc1", fc1);
  register_module("fc2", fc2);
}

torch::Tensor MyModel::forward(torch::Tensor x) {
  // conv1
  x = conv1->forward(x);
  x = torch::max_pool2d(x, 2);
  x = torch::relu(x);

  // conv2
  x = conv2->forward(x);
  x = conv2_drop->forward(x);
  x = torch::max_pool2d(x, 2);
  x = torch::relu(x);

  // fc1
  x = x.view({-1, 320});
  x = fc1->forward(x);
  x = torch::relu(x);

  // dropout
  x = torch::dropout(x, 0.5, is_training());

  // fc2
  x = fc2->forward(x);

  // log softmax
  x = torch::log_softmax(x, 1);

  return x;
}