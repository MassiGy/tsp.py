# TSP w/ Hubs.

### Members: 

- Massiles GHERNAOUT.
- Amine AIT MOKHTAR.

### Note: 
Me, Massiles, I'll take charge of the TSP w/ Hubs variant. Meanhwile, Amine will implement another.
The reason for this dispatch is due to the fact that we could not meet to work along together during these
last weeks as a duo.


### Description & Brainstorming: 


**Impact on the mathematical model:**

I choosed to tackle the Hub variant of the presented TSP problem.
The impact on the mathematical model is reduced to having a smaller vertex set, and thus a smaller arcs set.

In other words, the graph G will be reduced to a sub graph G' where G' = G \ W, W being the set of all nodes that
are the neighbors of a hub in G. (The hubs nodes stay in G', but not thier non-hub nieghbors)

This transformation (G->G') will be done before any resolution method gets invoked.

So the mathematical model will not change, it will just execute faster.

**The question becomes: How do we identify the Hubs ?**

Well, in this problem set, we do not have the arcs between the nodes. So the Hubs could not be identified by calculating the 
degrees distribution and then taking only those that are in the last quantile or so. That is not possible for this case.

But, we do have a routine that calculates the geometrical distance between all the points. One might say that a Hub is basically
a node that is at the center of a group of other nodes to which the distance does not exceed X% of the largest distance in the network.
X is to be defined.

i.e: 
The idea-rundown will be as follows: 
- we load our instance.
- we calculate the distances between all the node pairs.
- we keep track of the max and min distance.
- we map 20 or so (X=20%) from a 1-100 scale to a min.dist-max.dist scale. (just like the map function in Processing)
- So all the nodes that are at a center of a group (more then 3-4 nodes) of nodes to which the distance does not exceed that formula output, we mark them as Hubs.
- Then we reduce the graph to the set of hubs.


**Now the next question is: How do we identify THE HUB from the nodes that are within a local nodes concentration where the node2node distance does not exceed the target distance**

In other words, how to pick the Hub ?

Well....I do not really know.
But since the distance is here symetric (the graph is not a digraph), taking one randomly should do the trick. Or just order them in decending 
order and take the one that comes first in your subsequent traversals.

If the graph was a digraph, then the best fit will be the node that has the smallest distances sum to all the other nodes of the local concentration.
(Basically, it is the node that facilitate the transit the most.)

So, this should be a good start.



### Notes on the implementation and results:

So, as mentionned above, I've implemented the TSP w/ Hubs.
The routine that reduces the graph to a Hub only graph is called redhog (reduce to hub only graph).

This routine will:
- iterate over all the nodes.
- create a node to nodes struct that aggregate the neighbors of each node that are in a distance that less then a prefixed node2hub distance.
- sort this structure in decending order relative to the list of neighbors length.
- iterate over the sorted list, and pick the nodes that have more then a prefixed hubDegree count of neighbors.
- Then if the hubs overlap, one is a hub for another, merge the set of neighbors of the smaller cluster to the bigger one.
- return the new instance.

How this routine can be improved/relaxed ?:
- make the hub selection logic iterative and do not just merge the smaller clusters to the bigger ones with which they overalp.
- If there is an overlap, prefer cherrypicking the nodes that represent the overlap and add them to the closest hub neighbors list.
- If within the nodes that represent the overlap there is a hub, merge it as well and remove it from the hub list.
- update the hub list and the node2nodes struct along the way.
- Do this until there is no overlap.


**An overview on how the redhog routine reduces the graph to a hog:**

- Figure 1: the instance data plotted without any change.
- Figure 2: highlights all the potential hubs.
- Figure 3: highlights the reduced set of hubs.
- Figure 4: highlights the reduces set of hubs + outliers.
- Figure 5: the redhoged instance.

<br>

![redhog a graph of 20 nodes (data/instance1.txt)](./imgs/redhog_routine_demo_20nodes_graph.png)

redhog a graph of 20 nodes (data/instance1.txt)
<br>

---

![redhog a graph of 50 nodes (data/instance2.txt)](./imgs/redhog_routine_demo_50nodes_graph.png)

redhog a graph of 50 nodes (data/instance2.txt)
<br>

---

![redhog a graph of 100 nodes (data/instance3.txt)](./imgs/redhog_routine_demo_100nodes_graph.png)

redhog a graph of 100 nodes (data/instance3.txt)
<br>
<br>


**Remarks:**

So currently, my redhog routine is too aggressive, which is better for large data sets, but it might not be what is always needed.

Along side that, I've implemented some of the other resoultion routines/heurestics that were described in the handout of this problem set.

Then, there is the integration of the osmnx lib for the real world data and the visualization routines for better debugging and assessment.

On that, I've choosed to go with the charging_station data of Le Havre. See the ./notes.txt for the rationnal behind chosing an amenity.

Besides, I've added a switch to toggle the real-world-data mode and the local-data mode, which use diffrent instances and nodes data and 
diffrent plotting strategies as well. (see the main entry point of the TSP.py script.)

<br>


**Results of the TSP w/ Hubs resolution on the Le Havre's charging_stations network:**


<br>

![Le Havre's charging_stations network](./imgs/lh_charging_stations_network.png)

Le Havre's charging_stations network.
<br>

---


![Le Havre's charging_stations network redhoged](./imgs/lh_charging_stations_network_reduced_to_hubs_only.png)

Le Havre's charging_stations network reduced to hubs only.
<br>

---


![tsp_on_lh_charging_stations_network_with_trivial_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_trivial_resolution_strategy.png)

tsp_on_lh_charging_stations_network_with_trivial_resolution_strategy
<br>

---


![tsp_on_lh_charging_stations_network_with_random_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_random_resolution_strategy.png)

tsp_on_lh_charging_stations_network_with_random_resolution_strategy
<br>

---



![tsp_on_lh_charging_stations_network_with_compute_nearest_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_compute_nearest_resolution_strategy.png)

tsp_on_lh_charging_stations_network_with_compute_nearest_resolution_strategy
<br>

---


![tsp_on_lh_charging_stations_network_with_ilp_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_ilp_resolution_strategy.png)

tsp_on_lh_charging_stations_network_with_integer_lp_resolution_strategy
<br>

---

<br>

***Hurestics:***

<br>


![tsp_on_lh_charging_stations_network_with_multistart_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_multistart_resolution_strategy.png)

tsp_on_lh_charging_stations_network_with_multistart_resolution_strategy
<br>

---


![tsp_on_lh_charging_stations_network_with_multistart_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_multistart_resolution_strategy.png)
![tsp_on_lh_charging_stations_network_with_multistart_resolution_strategy_evolution](./imgs/tsp_on_lh_charging_stations_network_with_multistart_resolution_strategy_evolution.png)

tsp_on_lh_charging_stations_network_with_multistart_resolution_strategy and its evolution
<br>

---


![tsp_on_lh_charging_stations_network_with_multistart_with_2Ops_LS_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_multistart_with_2Opt_LS_resolution_strategy.png)
![tsp_on_lh_charging_stations_network_with_multistart_with_2Ops_LS_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_multistart_with_2Opt_LS_resolution_strategy_evolution.png)

tsp_on_lh_charging_stations_network_with_multistart_with_2Ops_LS_resolution_strategy and its evolution
<br>

---



![tsp_on_lh_charging_stations_network_with_multistart_with_Swap_LS_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_multistart_with_Swap_LS_resolution_strategy.png)
![tsp_on_lh_charging_stations_network_with_multistart_with_Swap_LS_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_multistart_with_Swap_LS_resolution_strategy_evolution.png)

tsp_on_lh_charging_stations_network_with_multistart_with_Swap_LS_resolution_strategy and its evolution
<br>

---


![tsp_on_lh_charging_stations_network_with_multistart_with_OrOpt_LS_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_multistart_with_OrOpt_LS_resolution_strategy.png)
![tsp_on_lh_charging_stations_network_with_multistart_with_OrOpt_LS_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_multistart_with_OrOpt_LS_resolution_strategy_evolution.png)

tsp_on_lh_charging_stations_network_with_multistart_with_OrOpt_LS_resolution_strategy and its evolution
<br>

---


![tsp_on_lh_charging_stations_network_with_tabou_search_resolution_strategy](./imgs/tsp_on_lh_charging_stations_network_with_tabou_search_resolution_strategy.png)
![tsp_on_lh_charging_stations_network_with_tabou_search_resolution_strategy_evolution](./imgs/tsp_on_lh_charging_stations_network_with_tabou_search_resolution_strategy_evolution.png)

tsp_on_lh_charging_stations_network_with_tabou_search_resolution_strategy and its evolution
<br>

---



### Comparaison of the results:

***Note:***
>Keep in mind that most hurestics have a max_iteration argument that I've set to the POIs_count². So in our instance it should be 90*90. (check the logs while executing.)
>
>It can be that more iterations can improve the result, but it should not shift the conclusions that will be derived from the visual that much since it will also increase computation time.

<br>

***How to read the visual of comparaison?***

To compare the results, I've choosed to go with a visualization that I've familiarized with in Finance. The idea of this visualization is to show which strategy has the 
best ratio distsum to computation_time. 

In Finances, this can be used to show which investment plan has the best ratio annual return on investement (ROI) to volatility/risk.

Since the smaller the distsum the better, I've inversed the y axis which represents the total distance of the TSP path so as the smallest appear at the top.

The x axis represent the computation time.

Which means, on the far top right, we have the most correct solutions but that take more computation time.

And at the far buttom left, we have the worst solutions but that take little to no computation time. 

The 50-50 tradeoff will be the center, of course.

<br>

![results comparaison on the TSP on Le Havre's charging_stations network with diffrent strategies.](./imgs/tsp_on_lh_charging_stations_network_resolution_strategies_comparaison.png)

Results comparaison on the TSP applied to Le Havre's charging_stations network with diffrent strategies.
<br>
<br>



i.e:
As you can see Integer LP is the best in terms of distsum, but it takes much more time then the hurestics. So the tradeoff is well shown with this visual.

From this visual we can conclude that:
- tabou search is really not a good resolution method, it takes more time then ilp and provides an equivalent result to most hurestics which are much much faster.
- random resolution method is the fastest but also can be the worst as in the this instance. (Trivial and random should always be worse but fast.)
- the other interseting point is that unless you have more then a second of computation time, you rather go with nearest neighbour, it is as fast as random and trivial and gives as good of a result compared to most hurestics.
