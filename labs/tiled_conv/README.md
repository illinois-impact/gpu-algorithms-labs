# Register-Tiled Neural Network Convolution Layer

## Objective
The goal of this lab is to implement the forward operation of a convolution layer using register-tiled matrix multiplication.

## CUDA Implementation

The skeleton code provides a basic CPU and GPU implementation of forward propagation for a convolutional layer.

Your CUDA implementation will be compared with the GPU version for the correctness at the end of each step for correctness and evaluated based on its achieved performance.

As mentioned in lecture, your code must work correctly in order to receive
credit.  You must implement register tiling to receive full functionality 
credit.  And your code must perform reasonably well (various student 
submissions as well as the instructor's version will be compared), as
some points will be awarded for performance.

You should implement your own register-tiled matrix-multiplication convolution written in CUDA.
Apply any optimization you think would bring benefit and feel free to modify any part of the code.
Once you have an optimized version, you may be interested in comparing your implementation with `cuBLAS` or `cuDNN`.

## Dataset Information

There is one set of convolution layer parameters assigned in this lab. We use random function to generate the input data images, hence, all the input test datasets between students and between each runs will be unique. Therefore, it is important to make sure that the output values match the results from the baseline GPU code. For computation simplicity, convolution filter weights are all set to 1.

* Input Dataset `N x C x H x W = N x 1 x 28 x 28`: all random variables
* Filter `M x C x K x K =  32 x 1 x 5 x 5`: all 1's

If you look at the `eval` calls at the bottom of `main.cu`, you will see 
an `ifdef` to switch between debugging, which uses N=20, and performance
measurement, which uses N=50,000.  Be sure that your code passes the
tests before turning it in (do so AFTER you have optimized performance as
well, just in case!).  Note that the performance case turns off verification
for speed, so you will not know that your code does not work if you
run only the performance version.


## Instructions

In the provided source code, you will find functions named `conv_forward_valid` and `conv_forward_kernel`.
These functions implement the sequential forward path and the GPU forward path of the convolution layer.
You do not have to modify these functions, they are used during validation.
You should modify the `conv_forward_opt_kernel` to implement your register-tiled kernel.
Don't forget to modify the host code in `convlayer_gpu_opt`.

You have to implement the host code to call GPU kernels, the GPU kernel functions and any additional CUDA memory management.
Once you have finished with CUDA implementation, you will be using the function `verify` to verify your solution with the results from the sequential code.
You will check the output feature maps, Y (or out), after the forward propagation.

You may optimize as much as you would like based on the layer parameters.
Do NOT optimize based on the weights all being set to 1.

## Range For Loops

Throughout the serial code, we use the [`range.hpp`][rangehpp] to make the code easier to understand. Essentially,

```{.cpp}
for (const auto ii : range(0, N)) {
    do_stuff(ii);
}
```

Is equivalent to

```{.cpp}
for (const auto ii = 0; ii < N; ii++) {
    do_stuff(ii);
}
```

The use of range introduces some overhead and you might get better speed if you remove it's usage.
