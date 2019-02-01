# Matrix Multiplication with Thread Coarsening and Register Tiling

## Objective 
The purpose of this lab is to practice the thread coarsening and register tiling optimization techniques using matrix-matrix multiplication as an example.

## Procedure 
\noindent \textbf{Step 1:} [Instructions on how to retrieve the new lab package.]
\\
\\
Edit the file `template.cu` to launch and implement a matrix-matrix multiplication kernel that uses thread coarsening and register tiling optimization techniques. The first input matrix has a column major layout and shall be tiled in the registers, the second input matrix has a row major layout and shall be tiled in shared memory, and the output matrix has a column major layout and shall be tiled in the registers. Macros have been provided to help you with accessing these matrices easily.

3. Test your code using rai

    `rai -p <path to your stencil folder>`

    Be sure to add any additional flags that are required by your course (`--queue` or others).

4. Submit your code on rai
