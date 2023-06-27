#include "my_dataset.h"

MyDataset::MyDataset(const std::string& data_root,
                     torch::data::datasets::MNIST::Mode phase)
    : mnist_dataset(torch::data::datasets::MNIST(data_root, phase)) {}