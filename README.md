# TSP w/ Hubs.

### Members: 

- Massiles GHERNAOUT.
- Amine AIT MOKHTAR.




### Description : 

We choosed to tackle the Hub variant of the presented TSP problem.

The impact on the mathematical model is reduced to having a smaller verticies set and thus also a smaller arcs set.

In other words, the graph G will be reduced to a sub graph G' where G' = G \ W, and W being the set of all nodes that
are the neighbors of the hubs nodes in G. (The hubs nodes stay in G', but not thier non-hub nieghbors)

This transformation (G->G') can be performed initially before any resolution method gets invoked.


**The question becomes: How do we identify the Hubs ?**

Well, in this problem set, we do not have the arcs between the nodes. So the Hubs could not be identified by calculating the 
degrees distribution and then taking only those that are in the last quantile or so. That is not possible for this case.

But, we do have a routine that calculates the geometrical distance between all the points. One might say that a Hub is basically
a node that is at the center of a group of other nodes to which the distance does not exceed X% of the largest distance in the network.
X is to be defined.

i.e: 
The rundown will be as follows: 
- we load our instance from our txt file.
- we calculate the distances between all the node pairs.
- we keep track of the max and min distance.
- we map 20 (X=20%) from a 1-100 scale to a min.dist-max.dist scale. (just like the map function in Processing)
- So all the nodes that are at a center of a group (more then 4 nodes) of nodes to which the distance does not exceed that formula output, we mark them as Hubs.
- Then we reduce the graph to the set of hubs.


**Now the next question is: How do we identify THE HUB from the nodes that are within a local nodes concentration where the node2node distance does not exceed the target distance**

In other words, how to pick the Hub ?

Well....I do not really know.
But since the distance is here symetric (the graph is not a digraph), taking one randomly should do the trick.
If the graph was a digraph, then the best fit will be the node that has the smallest distances sum to all the other nodes of the local concentration.
(Basically, it is the node that facilitate the transit the most.)

So, this should be a good start.
