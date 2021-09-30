# Triangle Counting

In this lab, you must modify template.cu to add two versions of 
neighbor-set-intersection-based triangle counting kernels.
The input to the kernel is the graph represented in unweighted 
adjacency matrix form, with directed edges based on some total ordering
of nodes.  Edge `(i,j)` is represented by a non-zero in row `i` and 
column `j` of the adjacency matrix.

Restricting edges to obey the total ordering of nodes produces a 
Directed Acyclic Graph (DAG), which lends to an efficient triangle 
counting approach, as discussed in lecture.  In DAG form, 
edge `(A,B)` forms the base of a triangle with nodes `A`,`B`,`C` 
if and only if edges `(A,C)` and `(B,C)` are also present in the graph. 

```
   C
  ^ ^
 /   \
A --> B
```

In this formulation, neither edge `(A,C)` nor `(B,C)` is considered to 
be the base of a triangle, so we count each triangle exactly once.

The graph representation used in this lab, based on the Pangolin library
from the IMPACT group led by Prof. Wen-mei Hwu, sorts all neighbor list 
arrays in increasing order of node number.  Given sorted neighbor lists,
neighbor set intersection can be performed using either a linear or a
binary search, as discussed in lecture.

## Step 1: Linear Search

Start by implementing a linear search in the `kernel_tc` kernel included
in the file `template.cu`.

Your kernel should parallelize over edges, and each thread should
* determine the source and destination node for the edge,
* use the row pointer array to determine the start and end of the neighbor list in the column index array, and
* use a linear search (see the slides in L8) to determine how many elements of those two arrays are common.

The edge destination array does not include a padding element at the end,
so the linear search pseudo-code given in the slides will need to be 
modified slightly to avoid out-of-bounds accesses. 

You will also need to write the kernel launch code for mode 1 in 
`count_triangles` to execute your kernel and to perform a reduction 
(either on the CPU or with another kernel on the GPU) over edges to 
determine the final triangle count.  ***Be sure that your counting 
(and reduction) kernels have finished before trying to do a
final summation on the CPU.***

Refer to the API information below for help in navigating the graph and 
vector structures provided for your use.

## Step 2: Separate the Linear Search Intersection

Once your linear search passes the first set of tests (the LINEAR launch in 
`rai_build.yml`), if you have not already done so, pull out the linear 
search intersection into a separate function for GPU execution (use the
`__device__` qualifier to avoid compilation on the host).  We suggest
passing in the array of edge destinations along with the starts and ends
of the two neighbor lists and returning the number of triangles found 
by the intersection, but the exact interface is left to you.

Be sure that your code still passes the tests.

## Step 3: Add Binary Search

Write a similar `__device__` function that uses binary search to find
the nodes in one neighbor list within the second neighbor list.  For 
this function, you may choose which neighbor list to walk linearly and
which to search, but you should then use the function correctly (so that
the longer list is the one being searched for elements of the shorter list).

Now add a second kernel similar to `kernel_tc` that uses your binary search
function instead of your linear search function.  Before calling binary
search, be sure to compare the size of the neighbor lists and swap them
if necessary so that the lists are in the correct order (whatever that 
order is for your binary search function, as discussed in the previous 
paragraph).  Finally, write launch code for this kernel under mode 2 in 
`count_triangles`.

Your code should now pass all tests for both launches in `rai_build.yml`, 
assuming that you implemented binary search and the new kernel launch 
correctly.

You may want to make some notes about timing.  Binary search is likely to
be slower than linear search for the sample graphs.

## Step 4: Choose the Right Search

As you may recall from our discussion in lecture, we expect binary search 
to be faster than linear search when the length of the shorter neighbor list 
is less than the length of the longer list divided by the log (base 2) of the
length of the longer list.  Play with the decision process until you find
something that you think works well.  In my brief experiments, I found
that I could beat both other approaches (using solely linear or solely
binary search) by using binary search when V was as least 64 and V/U was 
at least 6 (V is the longer list length, and U the shorter one).  
Have fun!  I expect to see some sort of dynamic selection
of search based on list length in your code, but anything reasonable is ok.

## Pangolin API reference

Pangolin provides a few useful data structures.
These structures are backed by CUDA unified memory, so you do not need to explicitly copy data between the host and device.

In your kernel implementation, you can access the graph through members of 
the Pangolin COOView object, which is
a hybrid **C**ompressed **S**parse **R**ow / **COO**rdinate format matrix.
The content is similar to a normal CSR matrix, but includes a row index for 
each non-zero (just as CSR has a column index for each non-zero).
Methods you may need include:

* `num_rows()` returns the number of rows
* `nnz()` returns the number of non-zero elements (edges of the graph)
* `row_ptr()` returns a pointer to the CSR row pointer array (length = `num_rows() + 1` or 0 if `num_rows() == 0`).
* `col_ind()` returns a pointer to the CSR column index array (length = `nnz()`).
* `row_ind()` returns a pointer to the CSR row index array (length = `nnz()`).

The COOView allows easy parallelization across all edges as well as easy 
traversal of edges from a particular node.

For example,
* the source and destination of edge `edgeId` are `row_ind()[edgeId]` and `col_ind()[edgeId]`, and
* the outgoing edges for node `i` are `col_ind()[row_ptr()[i]]` to `col_ind()[row_ptr()[i+1]]` (non-inclusive).

As the graph is unweighted, all non-zero values in the adjacency matrix are
equal to 1, thus there is no method to access the non-zero value for an edge.

You will also need to create Pangolin vectors to hold your triangle counts
(and possibly for your reduction results as well, should you choose to
perform the reduction on the GPU).

The two methods you need for the vector are as follows:

* `pangolin::Vector<ofWhatType> (size_t N, const_reference val)` is a 
constructor--your vectors should be initialized to the correct length and
initialized to 0 (passed as the second argument)
* `data()` returns a pointer to an array of `ofWhatType` in CUDA unified
memory, which can then be used on the host and passed to your kernels

## For Lab Developers (students can stop reading here)

Building this lab requires an installation of the [Pangolin](https://github.com/c3sr/pangolin) library and some graph data files.
Pangolin is used to read the graph data and provide the adjacency matrix to the students.
The lab should be configured with 
* `-DCMAKE_PREFIX_PATH` pointing to the Pangolin install so CMake can find the package.
* `-DGRAPH_PREFIX_PATH` set to the path that the graph data files are in (with escaped `\"`s), for example `-DGRAPH_PREFIX_PATH=\"/prefix/path\"`. In this example, a graph file would be at `/prefix/path/graph.bel`.

The Dockerfile for the lab `Dockerfile` is based off of the Docker images defined in github.com/c3sr/pangolin, with CMake added.
They were pushed to the docker hub with

```
docker build . -t raiproject/pumps2018:triangle-counting-amd64-cuda100
docker push raiproject/pumps2018:triangle-counting-amd64-cuda100
```


