#include <torch/torch.h>

#include <string>

#include "my_dataset.h"
#include "my_model.h"
#include "my_trainer.h"

int main() {
  // 超参数设置
  std::string data_root = "./data";
  int train_batch_size = 128;
  int test_batch_size = 1000;
  int total_epoch_num = 30;
  int log_interval = 10;
  int num_workers = 32;

  // 设置随机数种子
  torch::manual_seed(1);

  // 获取设备类型
  torch::DeviceType device_type = torch::kCPU;
  if (torch::cuda::is_available()) {
    device_type = torch::kCUDA;
  }
  torch::Device device(device_type);

  // 构造网络
  MyModel model;
  model.to(device);

  // 设置优化器
  torch::optim::SGD optimizer(model.parameters(),
                              torch::optim::SGDOptions(0.01).momentum(0.5));

  // 构造训练和测试dataset
  auto train_dataset =
      MyDataset(data_root, torch::data::datasets::MNIST::Mode::kTrain);
  auto test_dataset =
      MyDataset(data_root, torch::data::datasets::MNIST::Mode::kTest);

  // Trainer初始化
  auto trainer = MyTrainer(log_interval);
  for (size_t epoch = 1; epoch < total_epoch_num; ++epoch) {
    // 运行训练
    trainer.train(epoch, model, optimizer, device, train_dataset,
                  train_batch_size, num_workers);

    // 运行测试
    trainer.test(model, device, test_dataset, test_batch_size, num_workers);
  }
}