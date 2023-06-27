cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
add_definitions("-Wall -g")
set(CMAKE_PREFIX_PATH /home/geno/Documents/Projects/libtorch)
project(mnist_train)

# 需要找到Libtorch
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cpp my_model.cpp my_dataset.cpp my_trainer.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Libtorch是基于C++14来实现的
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
