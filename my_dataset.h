#pragma once
#include <torch/torch.h>

class MyDataset {
 public:
  MyDataset(const std::string& data_root,
            torch::data::datasets::MNIST::Mode phase);

 public:
  torch::data::datasets::MNIST mnist_dataset;
};