#include "my_trainer.h"

#include <torch/torch.h>

#include <cstdio>
#include <string>
#include <vector>

void MyTrainer::train(size_t epoch, MyModel& model,
                      torch::optim::Optimizer& optimizer, torch::Device device,
                      MyDataset& train_dataset, int batch_size,
                      int num_workers) {
  model.train();

  // 对MNIST数据进行Normalize和Stack（将多个Tensor stack成一个Tensor)
  auto dataset = train_dataset.mnist_dataset
                     .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                     .map(torch::data::transforms::Stack<>());

  // 构造 DataLoader, 设置 batch size 和 worker 数目
  auto data_loader =
      torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions()
                                                 .batch_size(batch_size)
                                                 .workers(num_workers));
  auto dataset_size = dataset.size().value();

  size_t batch_idx = 0;
  // 网络训练
  for (auto& batch : *data_loader) {
    // 获取数据和label
    auto data = batch.data.to(device);
    auto targets = batch.target.to(device);

    // 优化器 梯度清零
    optimizer.zero_grad();

    // 模型前向操作，得到预测输出
    auto output = model.forward(data);

    // 计算loss
    auto loss = torch::nll_loss(output, targets);

    // loss 反传
    loss.backward();
    optimizer.step();

    // 打印log信息
    if (batch_idx++ % log_interval_ == 0) {
      std::printf("\rTrain Epoch: %ld [%5llu/%5ld] Loss: %.4f", epoch,
                  batch_idx * batch.data.size(0), dataset_size,
                  loss.template item<float>());
    }
  }
}

void MyTrainer::test(MyModel& model, torch::Device device,
                     MyDataset& test_dataset, int batch_size, int num_workers) {
  // 测试时要将模型置为eval模式
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;

  // 对MNIST数据进行Normalize和Stack（将多个Tensor stack成一个Tensor)
  auto dataset = test_dataset.mnist_dataset
                     .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                     .map(torch::data::transforms::Stack<>());

  // 构造 DataLoader, 设置 batch size 和 worker 数目
  auto data_loader =
      torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions()
                                                 .batch_size(batch_size)
                                                 .workers(num_workers));
  auto dataset_size = dataset.size().value();

  for (const auto& batch : *data_loader) {
    // 获取数据和label
    auto data = batch.data.to(device);
    auto targets = batch.target.to(device);

    // 模型前向操作，得到预测输出
    auto output = model.forward(data);

    // 计算测试时的 loss
    test_loss += torch::nll_loss(output, targets,
                                 /*weight=*/{}, torch::Reduction::Sum)
                     .item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf("\nTest set: Average loss: %.4f | Accuracy: %.3f\n", test_loss,
              static_cast<double>(correct) / dataset_size);
}