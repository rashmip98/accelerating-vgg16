# Hardware Background

The goal of the hardware is to implement two GEMM convolution compute units on an FPGA. The key questions are asked in [question 5](../../lab4/docs/q5.md) in lab 4 and are worth adding to here:

* How to schedule operations (such as im2col and col2im) to maximize data reuse
* How to perform matrix multiplication with matrices that are larger than the available scratchpad memory

## Convolution as Matrix Multiplication

To accelerate convolution, we will be using GEMM. GEMM is an algorithmic transformation of regular direct convolution that works well systolic arrays due to its matrix-multiplication-inherent algorithm. Since markdown and static HTML is not the best format to teach about GEMM, we will yield to the lecture recordings. Although non-academic in nature, you may find the following articles helpful resources for understanding GEMM convolution: [Manas Sahni](https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/) and [Pete Warden](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/).

A decision to make is when and where to perform the mapping from a tensor to column form and vice versa with im2col and col2im. You are free to implement im2col and col2im in either hardware or software. Your decision is a good talking point in the report.

## Compute Units

In order to exploit parallelism inherent in the VGG16 model, you will create two compute units. How you use these compute units is a good talking point in your report. To create multiple compute units, you can tell the V++ linker to do so like below:

```
[connectivity]
nk=convole_fpga:2
```

More details can be found [here](https://github.com/Xilinx/Vitis_Accel_Examples/tree/master/host/mult_compute_units), as well as [here](https://github.com/ravicho/Vitis-AWS-F1-Developer-Labs/blob/master/modules/module_03/multi-CU.md).

## Hardware Specifications

To successfully complete the final project, you should meet the following requirements:

* Use IEEE 754 half-precision or single-precision floating-point (single-precision floating-point is recommended)
* Use less than or equal to 1024 DSPs per compute unit
* Implement GEMM convolution on FPGA and verify your code with vanilla conv2d and ensure the difference is within some epsilon
* Create and use all two compute units
