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

***You do not need to do this part of the lab for full credit.
If you choose to do it, be warned that (1) the debugging will be more
painful, and (2) it's probably as much or more work than the previous
parts, but it's only worth 20% as much--as extra credit.***

The last set of tests performs preprocessing on the GPU using three
additional kernels that you must write: `histogram`, `scan`, and `sort`.

Be sure to read the explanatory comments about the parameters that you
are given and the parallelization scheme for each kernel.

In `histogram`, your code must compute the number of input elements in each of 
the `NUM_BINS` bins.  Note that bins are equally sized, contiguous, 
and together cover the interval [0, `(grid_size - 1)` ).
The kernel is launched with a fixed number of thread blocks, so be sure
to process input data until each block runs out (using an appropriate
stride).

In `scan`, your code must perform an exclusive parallel scan on the
bin counts to produce an array of `(NUM_BINS + 1)` bin_ptrs.  The
last element should be equal to the number of input elements.  You should use
the Brent-Kung algorithm (see, for example, [lecture 16 of ECE408/CS483](http://lumetta.web.engr.illinois.edu/408-S20/slide-copies/ece408-lecture16-S20.pdf)).
The kernel is launched with one thread block and only half as many 
threads as bins (the right number for Brent-Kung).
The number of bins has been fixed to `1024` so that `scan` operations can be performed in a single thread block.

In `sort`, your code must sort the input elements by bin using the 
bin indices and counts that your other kernels computed previously.

Instructions about where to place each part of the code is
demarcated by the `//@@` comment lines.

