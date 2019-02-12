# Breadth First Search

## Objective 
The purpose of this lab is to understand hierarchical queuing in the context of the breadth first search algorithm as an example. You will implement a single iteration of breadth first search that takes aset of nodes in the current level (also called wave-front) as input and outputs the set of nodes belonging to the next level.

## Input Array
The input arrays input arrays and how they are use in the code

![image](assets/bfs.png "thumbnail")

## Input Informantion

The template code reads three input arrays:
 
nodePtrs is an integer array from 0 to numNodes that keep track of all the neighbor node pointers for each node in the graph. An example of using this array to loop through all of the neighbors of node number one you can do as follows:

    for (unsigned int nbrIdx = nodePtrs[1]; nbrIdx < nodePtrs[1]; ++nbrIdx) {
        ...
    }


nodeNeighbors is an integer array that keeps track of the index of all neighbors for all nodes. An example of how to get the index of all the neighbors for node number one and mark them as visited can be done by adding two lines to the above code:

    for (unsigned int nbrIdx = nodePtrs[1]; nbrIdx < nodePtrs[1]; ++nbrIdx) {
        unsigned int neighborID = nodeNeighbors[nbrIdx];
        nodeVisited[neighborID] = 1;
    }

currLevelNodes is also an integer array, from  0  to numCurrLevelNodes, which keep track of the current frontier.  This array is the starting point of your BFS. To loop through the current frontier can bedone as follows:

    for (unsigned int idx = 0; idx < numCurrLevelNodes; ++idx) {
        unsigned int node = currLevelNodes[idx];
    }

The image above shows how the arrays interact together to represent how nodes are linked together.

## Procedure 
1. Edit the file template.cu` to implement three versions of the BFS kernels: 

* Edit the kernel `gpu_global_queuing_kernel` in the file to implement the algorithm using just a global queue. Test by running ./bfs gq
* Edit the kernel `gpu_block_queuing_kernel` in the file to implement the algorithm using block and global queuing. Test by running ./bfs bq
* Edit the kernel `gpu_warp_queueing_kernel` in the file to implement the algorithm using warp, block, and global queuing. Test by running ./bfs wq

2. Test your code using rai

    `rai -p <bfs folder>`

    Be sure to add any additional flags that are required by your course (`--queue` or others).

3. Submit your code on rai
