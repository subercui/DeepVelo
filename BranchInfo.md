This branch is to build a MOCO dictionary on graph nets.

We need a way to constrain the 2-nd order derivatives, which relates to compare among neighbors.

1. Graph net wit laplasian can do that. 

2. But in large graphs, we want the minibatch of neighbors computed every time, not all the graph nodes. Also to input one cell instead a graph of cells is much more concise. So we want MOCO to handle all these concerns once and for all.

And currently this branch is implemented using DGL.
