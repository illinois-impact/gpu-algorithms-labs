# Triangle Counting

`rai_build.yml` is set up to test ONLY the required linear search kernel.
To enable testing of another optional kernel should you choose to write one, uncomment the line 
```
# - ./tc -c OTHER
```

For this lab, you will modify template.cu to add a neighbor-set-intersection-based triangle counting kernel.
The input to the kernel is the graph represented in unweighted adjacency matrix form, with directed edges based on some [total ordering](https://en.wikipedia.org/wiki/Total_order) of nodes.
Edge `(i,j)` is represented by a non-zero in row `i` and column `j` of the adjacency matrix.

Thanks to the total ordering of nodes, this forms a Directed Acyclic Graph (DAG), which lends to an efficient triangle counting approach.
In this DAG form, we can consider edge `(B,C)` to form the base of a triangle with nodes `A`,`B`,`C` if and only if there is an edge `(B,A)`, and edge `(C,A)`. 

```
   A
  ^ ^
 /   \
B --> C
```

Note that in this formulation, neither edge `(B,A)` or `(C,A)` is considered to be the base of the triangle, so we do not double- (or triple-) count any triangles.

The neighbor set intersection approach is as follows.

```
prodecture TriCount(edge):
  { nSrc } = neighbors(edge.src) // set of neighbors of edge source (B)
  { nDst } = neighbors(edge.dst) // set of neighbors of edge destination (C)
  return size(nSrc ∪ nDst)   // size of union of neighbor sets (all the different As)
```

Your kernel can operate on an edge:
* Determine the source and destination node for the edge
* use the row pointer array to determine the start and end of the neighbor list in the column index array
* determine how many elements of those two arrays are common

Our graph representation sorts all neighbor list arrays in increasing order.
You can use this fact to accelerate the search:
* REQUIRED: start with a pointer to the beginning of each array, and increment whichever pointer points to a smaller value. If the two pointers are the same, then there is an element in common and both should be incremented.
* OPTIONAL: use a binary search in one array to check for the existence of elements from the other array.

After determining the triangle count for each edge, a global reduction over edges can be performed to determine the final triangle count.

## Pangolin API reference

Pangolin provides a few useful data structures.
These structures are backed by CUDA unified memory, so you do not need to explicitly copy data between the host and device.

In your kernel implementation, you can access the graph through members of the COOView object.
This is a hybrid **C**ompressed **S**parse **R**ow / **COO**rdinate format matrix.
It is similar to a normal CSR matrix, but agumented with a row index for each non-zero (just like how CSR has a column index for each non-zero).
See the API reference for the COO view below. In short:

* `num_rows()` returns the number of rows
* `nnz()` returns the number of non-zeros
* `row_ptr()` returns a pointer to the CSR row pointer array (length = `num_rows() + 1` or 0 if `num_rows() == 0`).
* `col_ind()` returns a pointer to the CSR column index array (length = `nnz()`).
* `row_ind()` returns a pointer to the CSR row index array (length = `nnz()`).

This is to allow easy parallelization across all edges, or easy traversal of edges from a particular node.

For example:
* The source and destination of an edge are therefore `row_ind()[edgeId]` and `col_ind()[edgeId]`.
* The outgoing edges for node `i` are `col_ind()[row_ptr()[i]]` to `col_ind()[row_ptr()[i+1]]` (non-inclusive).

Note that there is no way to actually access the non-zero value.
As the graph is unweighted, the existence of a non-zero implies that the value is 1.

A partial [Pangolin API reference](https://zealous-heyrovsky-d34b6e.netlify.com/annotated.html) is available.

| API | Link |
|-|-|
| COO View class | [direct link](https://zealous-heyrovsky-d34b6e.netlify.com/classcooview) |
| Vector class | [direct link](https://zealous-heyrovsky-d34b6e.netlify.com/classpangolin_1_1vector)


## For Lab Developers (students can stop reading here)

Building this lab requires an installation of the [Pangolin](github.com/c3sr/pangolin) library and some graph data files.
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


