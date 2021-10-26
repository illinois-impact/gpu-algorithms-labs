# Parallel Merge

### Objective
The parallel merge operation is an important component of parallel sorting 
algorithms. In this lab, you'll write kernels for merging two sorted arrays.

![image](assets/merge.png "merge operation")

An ordered merge function takes two sorted arrays, `A` and `B`, and merges 
them into a single sorted array, `C`.   All versions of the kernel require
five arguments: the arrays `A`, `B`, and `C`, and the array lengths `A_len` 
and `B_len`.  The length of array `C` is the sum of the other two lengths;
`C` is allocated before calling any of the kernels.

You are required to implement both a basic parallel merge and a 
tiled parallel merge, and may choose to implement a circular buffer merge:
 - **`gpu_basic_merge` (REQUIRED)**: Each thread is responsible for an 
output range. Each thread then uses the `co-rank` function to determine 
the two input ranges that the thread is responsible for merging.  Once 
input and output ranges are identified, each threads independently 
performs a sequential merge in parallel.

 - **`gpu_tiled_merge` (REQUIRED)**: This kernel must use shared memory to 
improve both reuse and memory coalescing relative to the first kernel. 
In this kernel, input and output ranges are divided at the thread block 
level. Threads in a block then collaboratively load inputs to shared memory 
and perform the merge operation. This approach reduces the performance
penalty imposed by the irregular memory access pattern in the basic merge 
kernel.

 - **`gpu_circular_buffer_merge` (OPTIONAL)**: As a further optimization,
one can use a circular buffer to increase the utilization of shared memory. 
The general approach similar to that used in the tiled merge kernel, but 
the `co_rank` and `sequential_merge` functions must be changed to support 
the circular buffer. The coding complexity thus increases significantly. 
Implementing this kernel (and supporting code) correctly is thus worth
2 points of extra credit.

The template provided contains a sequential merge function, `merge_sequential`
as well as a `ceil_div` function that finds the ceiling of the quotient of 
two integers.  You must implement the `co_rank` function yourself, and must
produce a second version should you choose to implement the circular buffer
version of the kernel.

### Running the code
By default, `rai_build.yml` builds and runs all three kernels 
and checks the results against the reference solution.  The last
set of tests uses the extra credit kernel (`gpu_circular_buffer_merge`), 
so you can earn full points without passing those tests.  As always,
of course, your code will be examined for bugs that do not show up in 
the tests.

