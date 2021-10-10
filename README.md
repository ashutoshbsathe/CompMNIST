# CompMNIST [WIP]
ComplexMNIST dataset for complex vision problem

# Usage
To generate the dataset, you need to have `potrace` and `svgpathtools` installed and available in PATH. By default, everything is tested on Linux (windows compatibility soon)
* Clone the repo
* Go to `src/` folder. Every next reference will be relative to this directory
* Download MNIST dataset (the raw files) into `./mnist_raw` directory
* Run `sort_mnist.py`
* Run `trace_mnist.py`
* Run `new_complex_mnist.py`

TODO: Use [Numba](https://www.youtube.com/watch?v=x58W9A2lnQc) for making your code extremely fast. [Cython](https://www.youtube.com/watch?v=8DuyATDaIdM) might be good too
