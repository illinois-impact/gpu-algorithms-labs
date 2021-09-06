# Matrix Multiplication with Thread Coarsening and Register Tiling

## Objective 

The purpose of this lab is to practice the thread coarsening and register tiling optimization techniques using matrix-matrix multiplication as an example.

The code given in lecture is a rough guide for doing so, but you will
need to add conditionals and other code as necessary to ensure that
the kernel executes properly and without illegal memory accesses.
Specifically, **you may not make assumptions about the size of the input
matrices**.  

## Procedure 

### Step 1: 

Edit the file <code>template.cu</code> to launch and implement a 
matrix-matrix multiplication kernel that uses thread coarsening and 
register tiling optimization techniques. The first input matrix has 
a column major layout and should be tiled in the registers.  The 
second input matrix has a row major layout and should be tiled in 
shared memory, and the output matrix has a column major layout and 
should be tiled in the registers. Macros have been provided to 
help you with accessing these matrices easily.

Note that you must also write the kernel launch--your kernel will not
execute until you do so.

### Step 2:

Test your code using rai

<code>rai -p \<path to your stencil folder\></code>

The testing provided will execute your code with a variety 
of input matrix dimensions.

### Step 3:

Your last RAI submission will be used for grading.  Be sure that it
passes all tests for full points (you may still lose points for bugs
not exposed during testing).

