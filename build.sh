mkdir build
cd build
cmake ..  -DCMAKE_PREFIX_PATH=/home/geno/Documents/Projects/libtorch && make -j8
mkdir data && cd data
wget "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" && gunzip train-images-idx3-ubyte.gz
wget "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" && gunzip train-labels-idx1-ubyte.gz
wget "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" && gunzip t10k-images-idx3-ubyte.gz
wget "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz" && gunzip t10k-labels-idx1-ubyte.gz
