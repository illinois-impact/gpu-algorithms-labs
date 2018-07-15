## Objective

The purpose of this lab is to get you familiar with the scatter pattern.

## Instructions

In the provided source code you will find a function named `s2g_cpu_scatter`.
This function implements a simple scatter pattern on CPU.
It loops over an input array, then for each input element it performs some computation (`outInvariant(...)`), loops over the output array, does some more computation (`outDependent(...)`), and accumulates to the output element.

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.
