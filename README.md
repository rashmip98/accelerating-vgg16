# Project: Accelerating VGG16 DCNN with an FPGA

## Usage
To run the code, run the benchmark.ppi file. It will take roughly 2 hours to complete the run.

## Description
The final report for this project can be found [here](https://drive.google.com/file/d/1_t-ZUs8Ksm7X3zX2y6vlzE8V9R8Q-tjK/view). 
For this project, we first began by implementing a General Matrix Multiply (GEMM) convolution kernel on an AWS F1 FPGA. Then, we integrated the kernel into PyTorch using C++ extension. Finally, we searched for an efficient scheduling of our computation and benchmarked our results.

The overall goals of the project were:

* To create and verify a functional GEMM convolutional kernel on FPGA
* Implement two compute units implementing our kernel
* Optimize our GEMM kernel 
* Use C++ Extension to communicate to our kernel with PyTorch API
* Implement software baseline and verify benchmarking with one compute unit
* Schedule computation and benchmark results with all two compute units

## Files
The main files for the project are located in project/src/
