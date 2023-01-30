# 7-point Stencil with Thread-coarsening and Register Tiling

## Objective
The purpose of this lab is to practice the thread coarsening and register tiling optimization techniques using 7-point stencil as an example.

## Procedure
1. Edit the `kernel` function in `template.cu` to implement a 7-point stencil (refer to the [lecture slides](http://lumetta.web.engr.illinois.edu/508/slides/lecture3-x4.pdf)) with combined register tiling and x-y shared memory tiling, and thread coarsening along the z-dimension.

    ```
    out(i, j, k) =  C0 *in(i, j, k)
                  + C1 * (  in(i-1, j, k)
                          + in(i, j-1, k)
                          + in(i, j, k-1)
                          + in(i+1, j, k)
                          + in(i, j+1, k)
                          + in(i, j, k+1) )
    ```

2. Edit the `launchStencil` function in `template.cu` to launch the kernel you implemented. The function should launch 2D CUDA grid and blocks, where each thread is responsible for computing an entire column in the z-deminsion.

    `A0` and `Anext` in the code template correspond to `in` and `out`, respectively. The output dimension of the 7-point stencil computation is one smaller than the input dimension on both sides for all boundaries (e.g., output dimension is 6x6x6 for an input of 8x8x8). Only those "internal" elements needs to be calculated.

3. Test your code using rai

    `rai -p <path to your stencil folder>`

    Be sure to add any additional flags that are required by your course (`--queue` or others).

4. Submit your code on rai

## Other notes

To simplify the kernel code, you do not need to support input data with z-extent less than 2.

The data is stored in column-major order. For example, you might consider using a macro to simplify your data access indexing:

```c++
__global__ void kernel(...) {}
    #define A0(i, j, k) A0[((k)*ny + (j))*nx + (i)]
    // your kernel code
    #undef A0
}
```
