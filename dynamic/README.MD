Option 1:
    Unoccupied local GPU, now I want to run
Option 2:
    New thing on a new node GPU.

We need to understand what information we need to do this.


Worst case: I need a resource on a node and it's on another node.




For patch sending:
- A node can hold multiple patches. A neighbouring patch is therefore
    1) On the node
        a) contigiously located next to the patch.
        b) somewhere else in memory.
    2) Off the node

For case 1a, array access can simply be done in the negatives and size + 1, and neighbours can be accessed.
For case 1b, neighbours have to be copied over from a location.
For case 2, neighbours have to be communicated over MPI.

See tasks.cpp 'fetch_and_update_neighbours'.


A task object holds as input the patch size + neighbours (in this case vector length + 2), and as output simply the patch size.
For more efficient buffer swapping, output length can equals the input length, but in this case we just copy over the results back to the input (slow, but works).

Biggest question is how to update neighbouring patches that the patch has been split.
    Start: Node 1 has a patch, neighboured by Node 2.
    1) Node 1 splits its patch, such that the new patch is now located on node 3, and the new patch now neighbours node 2.
        Patches are located in a (1 - 3 - 2) fashion.
    2) Node 1 and node 3 have communication; node 1 previously woke up / notified node 3, and 1 informs 3 of the new task, and splits its own task locally.
    3) ...
    4) Node 1 or 3 somehow notifies node 2 of the change.
