## Objective

The purpose of this lab is to help you understand input binning and its 
impact on performance in the context of a simple computation example. 

Your task is to compute a value for each point in a 1-dimensional grid
of points.  Grid point coordinates are integers ranging from 0 to 
`(grid_size - 1)`.  You are also provided with a set of input elements,
which are located at real-valued (`float`) coordinates **within the range 
[0,`grid_size`]**.  Each input element also has a real value (a `float`).
The illustration below depicts the grid points and input elements.

![image](assets/fig.png "thumbnail")

The value that you must compute for each grid point
is a sum over all input elements, and is defined as follows:

![image](assets/formula.png "thumbnail")

where out[j] is the value computed for the grid point at coordinate j,
the sum is over all N input elements, val[i] is the value of input
element i, and pos[i] is the position of input element i.

## Instructions

Implement the kernels in `main.cu` in the following order: `gpu_normal_kernel`,
`gpu_cutoff_kernel`, and `gpu_cutoff_binned_kernel`.

Be sure to read the explanatory comments about the parameters that you
are given and the parallelization scheme (one thread per grid point/gather, 
without coarsening).

In `gpu_normal_kernel`, your code should include all input elements' effects
on all grid points.

In `gpu_cutoff_kernel`, your code should include only those input elements
with distance from the grid point **strictly less than** the specified 
cutoff distance (given as the square, `cutoff2`). 

In `gpu_cutoff_binned_kernel`, the input elements are binned: both the
values and positions are sorted in increasing order of position, and
an array of bin indices (of length `(NUM_BINS + 1)`) is provided to
define the index ranges for each bin.  Your code should **first compute the
bins that overlap the cutoff region**, then look at all input elements within
those bins.  Looping over all bins will not earn full credit.

Initial tests perform preprocessing for `gpu_cutoff_binned_kernel` 
on the CPU.

### Binning on the GPU

The last set of tests performs preprocessing on the GPU using three
additional functions that you must write.

.  You must edit the `histogram`, `scan`, and `sort` kernels to perform the preprocessing on the GPU. The number of bins has been fixed to `1024` so that `scan` operations can be performed in a single thread block.

Instructions about where to place each part of the code is
demarcated by the `//@@` comment lines.

