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




