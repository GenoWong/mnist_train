#pragma once
#include <torch/torch.h>

#include "my_dataset.h"
#include "my_model.h"

class MyTrainer {
 public:
  MyTrainer(int log_interval) : log_interval_(log_interval){};

  void train(size_t epoch, MyModel& model, torch::optim::Optimizer& optimizer,
             torch::Device device, MyDataset& train_dataset, int batch_size,
             int num_workers);

  void test(MyModel& model, torch::Device device, MyDataset& test_dataset,
            int batch_size, int num_workers);

 private:
  int log_interval_;
};