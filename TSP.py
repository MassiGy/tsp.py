# -*- coding: utf-8 -*-

import random
import math
import itertools
import time
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD
from pathlib import Path

import osmnx as ox

def mapval(val, inMin, inMax, outMin, outMax):
    return outMin + ((float)(val-inMin)/(inMax-inMin)) * (outMax-outMin)




class Sommet:
    cpt_sommet = 0
    
    def __init__ (self, x, y):
        self.id = Sommet.cpt_sommet
        Sommet.cpt_sommet += 1
        
        self.x = x
        self.y = y
    
    def getId (self):
        return self.id
    
    def setId (self, id):
        self.id = id
    
    def getX (self):
        return self.x
    
    def getY (self):
        return self.y
    
    def str(self):
        return '({}:{:.2f},{:.2f})'.format(self.id,self.x,self.y)
    

    def affiche (self):
        print(self.str())




class Instance:
    def __init__ (self, name, n, sommets=[]):
        self.name = name
        self.nb_sommets = n
        self.sommets = sommets
        
        self.dist = [[0.0] * self.nb_sommets for _ in range(self.nb_sommets)]
        #self.init()
    
    def size (self):
        return self.nb_sommets
    
    def init (self):
        assert self.sommets == [] # to not override
        self.generateNodes()
        self.computeDistances()

    def generateNodes (self):
        self.sommets = [Sommet(random.uniform(0,100), random.uniform(0,100)) for i in range(self.nb_sommets)]

    @classmethod
    def fromFile(classref, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        name = str(filename).removesuffix(".txt")
        nb_sommets = int(lines[0].strip())
        instance = classref(name, nb_sommets)
        instance.sommets = []
        

        rest = lines[1:]
        for line in rest:
            parts = line.strip().split()

            # format should be [x y]
            if len(parts) != 2:
                continue

            x, y = float(parts[0]), float(parts[1])
            s = Sommet(x, y)
            instance.sommets.append(s)
        

        instance.computeDistances()
        return instance
    
    def computeDistances (self):
        self.maxdist = -float('inf')
        self.mindist = float('inf')
 
        
        for si in self.sommets:
            for sj in self.sommets:
                siid, sjid= si.getId(),sj.getId()

                delta_x = si.getX() - sj.getX()
                delta_y = si.getY() - sj.getY()
                self.dist[siid][sjid] = math.sqrt(delta_x ** 2 + delta_y ** 2)

                dist = self.dist[siid][sjid]
                
                # this is the init
                if self.maxdist == -1.0:
                    self.maxdist = dist
                
                if self.mindist == -1.0:
                    self.mindist = dist

                if self.maxdist < dist:
                    self.maxdist = dist
            
                if self.mindist > dist and siid != sjid:
                    self.mindist = dist

    # reduce to hubs only graph
    def redhog(self, hubradius):
        
        print(f"[metrics.redhog]: Reduce graph to hubs only graph, input graph # nodes={self.nb_sommets}")
        print(f"[metrics.redhog]: maxdist={self.maxdist}, mindist={self.mindist}")
        toHubDist = mapval(hubradius, 0, 100, self.mindist, self.maxdist)
        print(f"[metrics.redhog]: Node To Hub Distance Radius={toHubDist}")
        
        hubs:list[Sommet] = []
        hubdeg = 2
        print(f"[metrics.redhog]: Hub Min Degree={hubdeg}")
       
        # these are the nodes that are not wihtin a concentration of nodes (no hubs nearby)
        outliers: list[Sommet] = []
        
        # node2nodes will keep track of all the nodes that are within a distance
        # d < toHubDist of each node n
        n2ns:dict[Sommet, list[Sommet]] = dict()

        # iterate through the distances
        for si in self.sommets:
            n2ns[si] = []
            for sj in self.sommets:
                siid, sjid= si.getId(),sj.getId()
                
                if siid == sjid:
                    continue

                if self.dist[siid][sjid] <= toHubDist:
                    n2ns[si].append(sj)
                
        # Find hub candidates, IMPORTANT: it is crucial to sort them
        # by degree in a decending order, this way the selection will 
        # be much more suitable for the end result (reduced graph)
        candidates = sorted(n2ns.items(), key=lambda item: -len(item[1]))
        for n, neighbors in candidates:
            if len(neighbors) >= hubdeg and n not in hubs:
                hubs.append(n)
                
                
        Instance.visualize_hubs(n2ns, hubs, name="(All potential hubs)")
        
        noChange = False
        while noChange == False:        # reduce iteratively until g converges to a given state.
            noChange = False
            # reduce the overlaps 
            to_remove = set()
            for si in hubs:
                for sj in hubs:
                    siid, sjid= si.getId(),sj.getId()
                    
                    if siid == sjid or self.dist[siid][sjid] > toHubDist:
                        continue

                    l1 = len(n2ns[si])
                    l2 = len(n2ns[sj])

                    # IMPORTANT! since our hubs are sorted by degree in a 
                    # desending order, prefer pushing to si's cluster first
                    # thus the >= rather then just > is super important (3 hours for just that !)
                    if l1 >= l2:
                        n2ns[si] = list(set(n2ns[si]) | set(n2ns[sj]))
                        to_remove.add(sj)
                    else: 
                        n2ns[sj] = list(set(n2ns[sj]) | set(n2ns[si]))
                        to_remove.add(si)

                    if si in n2ns[si]: n2ns[si].remove(si)
                    if sj in n2ns[sj]: n2ns[sj].remove(sj)

            
            noChange =  len(to_remove) == 0
            hubs = [h for h in hubs if h not in to_remove]

        
        Instance.visualize_hubs(n2ns, hubs, name="(Reduced hubs set - merging)")     
        
        print(f"[metrics.redhog]: # hubs in input graph={len(hubs)}")

        hubsClustersNodes:list[Sommet] = hubs.copy()
        for n in hubs:
            hubsClustersNodes.extend(n2ns[n])
        
        
        hubsClustersNodes = set(hubsClustersNodes)
        
        outliers = list(set(self.sommets) - set(hubsClustersNodes))
        print(f"[metrics.redhog]: # outliers in input graph={len(outliers)}. About to add them to the reduced graph nodes list.")
        
        hubs.extend(outliers)


        for i in range(len(hubs)):
            # update the ids for the hubs
            n = hubs[i]
            n.setId(i)


        Instance.visualize_hubs(n2ns, hubs, name="(Hubs+Outliers - Final reduced Graph)")     
            

        # now that we have the hubs, create a new instance
        # using those hubs as nodes
        hoginst = Instance(str("redhog "+self.name), len(hubs), hubs)
        hoginst.computeDistances()
        print(f"[metrics.redhog]: Reduced graph # nodes={len(hubs)}")

        # show all our visualizations
        plt.show()

        return hoginst
    
    @classmethod
    def visualize_hubs(cls, n2ns: dict, hubs: list, name=""):
        plt.figure()
        
        # Plot all nodes
        for node in n2ns:
            plt.scatter(node.x, node.y, color='grey', s=30, zorder=1)
        
        # Highlight hubs and their connections
        for hub in hubs:
            plt.scatter(hub.x, hub.y, color='red', s=100, zorder=3)
            
            for connected_node in n2ns[hub]:
                plt.plot([hub.x, connected_node.x], [hub.y, connected_node.y], 
                        color='blue', linewidth=1, zorder=2)

        plt.title("Hub-Node Connections " + str(name))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis("equal")
        plt.draw()

    def affiche (self):
        print('{} sommets: '.format(self.nb_sommets))
        for s in self.sommets:
            print('  ', end='')
            s.affiche()
        print('dist:')
        for line in self.dist:
            for elt in line:
                print(' {:6.2f}'.format(elt), end='')
            print()

    def plot (self):
        plt.figure()
        plt.title('instance {}: {} sommets'.format(self.name, self.nb_sommets))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-1,101)
        plt.ylim(-1,101)
        x = [elt.getX() for elt in self.sommets]
        y = [elt.getY() for elt in self.sommets]
        plt.scatter(x,y)
        plt.grid(True)
        plt.draw()


class Solution:
    def __init__ (self, inst, desc):
        self.instance = inst
        self.name = desc
        self.sequence = [-1] * inst.size()
        self.valeur = 0.0
        self.temps = 0.0

    def getSequence (self)->list:
        return self.sequence

    def getValeur (self)->float:
        return self.valeur
    
    def getTemps (self):
        return self.temps
    
    def setTemps (self, t):
        self.temps = t

    def setSequence (self, s):
        self.sequence = s
        self.evalue()
    
    def evalue (self):
        val = 0.0
        for i in range(-1, len(self.sequence)-1):
            val += self.instance.dist[self.sequence[i]][self.sequence[i+1]]
        self.valeur = val
        
    def affiche (self):
        print('solution \'{}\': {} -> val = {:.6f} temps = {:.6f} s'.format(self.name, self.sequence, self.valeur, self.temps))

    def plot (self):
        plt.figure()
        plt.title('\'{}\': valeur = {:.2f} en {:.2f} s'.format(self.name, self.valeur, self.temps))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-1,101)
        plt.ylim(-1,101)
        x = [self.instance.sommets[index].getX() for index in self.sequence]
        y = [self.instance.sommets[index].getY() for index in self.sequence]
        x.append(x[0])
        y.append(y[0])
        plt.plot(x, y, marker='o')
        for index in self.sequence[1:]:
            sommet = self.instance.sommets[index]
            plt.text(sommet.getX(), sommet.getY() + 2, str(sommet.getId()))
        plt.text(x[0], y[0] + 2, str(self.instance.sommets[self.sequence[0]].getId()), color = 'r')
        plt.grid(True)
        plt.show()


class Heuristiques:
    def __init__ (self, inst: Instance):
        self.instance = inst
        self.evolution = []

    def plot_evo (self):
        plt.figure()
        plt.title('evolution')
        plt.xlabel('temps (s)')
        plt.ylabel('valeur')
        x = [elt[0] for elt in self.evolution]
        y = [elt[1] for elt in self.evolution]
        plt.plot(x,y, marker='o')
        plt.grid(True)
        plt.show()
        
    def compute_triviale (self):
        seq = [i for i in range(self.instance.size())]
        s = Solution(self.instance, 'triviale')
        s.setSequence(seq)
        return s
        
    def compute_random (self):
        seq = [i for i in range(self.instance.size())]
        random.shuffle(seq)
        s = Solution(self.instance, 'random')
        s.setSequence(seq)
        return s
    
    def compute_nearest (self):
        available = [i for i in range(1,self.instance.size())]
        current = 0
        seq = [current]
        
        while len(available) != 0:
            best = None
            dist = float('inf')
            for elt in available:
                if self.instance.dist[current][elt] < dist:
                    dist = self.instance.dist[current][elt]
                    best = elt
            seq.append(best)
            available.remove(best)
            current = best
        
        s = Solution(self.instance, 'nearest neighbour')
        s.setSequence(seq)
        return s

    def compute_enumerate (self):
        record = Solution(self.instance, 'enumerate')
        
        if self.instance.size() > 11:
            print('-> too many nodes for the enumeration. Stop.')
            return record

        debut = time.time()
        seq = [i for i in range(self.instance.size())]
        record.setSequence(seq)
        
        s = Solution(self.instance, 'tmp')
        perm = itertools.permutations(seq)
        for p in perm:
            s.setSequence(list(p))
            if s.getValeur() < record.getValeur():
                record.setSequence(s.getSequence())
        duree = time.time() - debut
        record.setTemps(duree)
        return record

    def ilp(self):  
        record = Solution(self.instance, 'Integer LP')

        n = self.instance.nb_sommets
        prob = LpProblem('TSP', LpMinimize)
        x = LpVariable.dicts('x', (range(n), range(n)), cat='Binary')
        u = LpVariable.dicts('u', range(n), lowBound=0, upBound=n-1, cat='Integer')

        #fonction objectif
        prob += lpSum(
            self.instance.dist[i][j] * x[i][j]
            for i in range(n)
            for j in range(n)
        )

        # Contraintes

        #degré
        for i in range(n):
            #sortie unique de chaque ville
            prob += lpSum(x[i][j] for j in range(n) if j != i) == 1
            #enter unique de chaque ville
            prob += lpSum(x[j][i] for j in range(n) if j != i) == 1

        #élimination des sous-tour
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    prob += u[i] - u[j] + n * x[i][j] <= n - 1

        prob.solve(PULP_CBC_CMD(msg=False))

        #Reconstuire le tour
        tour = [0]

        # On n’a pas listé n sommet on suit  l'arc choisi pour passer à la ville suivante
        while len(tour) < n:
            i = tour[-1]
            for j in range(n):
                if i != j and x[i][j].value() >= 0.99:
                    tour.append(j)
                    break
        
        # refermer le cycle
        tour.append(0)

        record.setSequence(tour)
        return record

    def mvt2Opt (self, s: Solution):
        seq = s.getSequence()
        s.affiche()
        dist = self.instance.dist
        for i in range(0,len(seq)-2):
            for j in range(i+1, len(seq)-1):

                if i == 0 and j == len(seq)-2:
                    continue
                
                delta = dist[seq[i-1]][seq[i]] + dist[seq[j]][seq[j+1]] - dist[seq[i-1]][seq[j]] - dist[seq[i]][seq[j+1]]
                
                if delta > 0:
                    while i<j:
                        seq[i],seq[j] = seq[j], seq[i]
                        i += 1
                        j -= 1
                    
                    s.setSequence(seq)
                    s.affiche()
                    return True
        return False
    
    def OrOpt (self, s: Solution, nb_iter=20):
        s.affiche()
        refseq = s.getSequence()
        
        seq = refseq.copy()
        _s = Solution(Instance("", len(refseq)), None)   # shallow struct (just for the eval fn) 
        

        counter = 0
        while counter <= nb_iter:
            indx_removal = random.randint(0, len(seq)-1)
            node = seq.pop(indx_removal)
            indx_insertion = random.randint(0, len(seq)-1)
            seq.insert(indx_insertion, node)

            _s.setSequence(seq)

            if _s.getValeur() < s.getValeur():
                s.setSequence(seq)
                s.affiche()
                return True
        
            counter +=1

        return False

    def swap (self, s: Solution, nb_iter=20):
        s.affiche()
        refseq = s.getSequence()

        seq = refseq.copy()
        _s = Solution(Instance("", len(refseq)), None)   # shallow struct (just for the eval fn) 
        

        counter = 0
        while counter <= nb_iter:
            fst = random.randint(0, len(seq)-1)
            snd = random.randint(0, len(seq)-1)
            seq[fst], seq[snd] = seq[snd], seq[fst]

            _s.setSequence(seq)

            if _s.getValeur() < s.getValeur():
                s.setSequence(seq)
                s.affiche()
                return True
        
            counter +=1

        return False

    def multistart (self, nb_iter = 20):
        record = Solution(self.instance, 'Multistart')
        debut = time.time()
        self.evolution = []
        for iter in range(nb_iter):
            s = self.compute_random()
            if record.getValeur() == 0.0 or s.getValeur() < record.getValeur():
                record.setSequence(s.getSequence())
                duree = time.time() - debut
                self.evolution.append((duree,s.getValeur()))
        return record
    
    def multistart_LS_2Opt (self, nb_iter = 20):
        record = Solution(self.instance, 'Multistart_LS_2Opt')
        debut = time.time()
        self.evolution = []
        for iter in range(nb_iter):
            s = self.compute_random()
            self.mvt2Opt(s)
            if record.getValeur() == 0.0 or s.getValeur() < record.getValeur():
                record.setSequence(s.getSequence())
                duree = time.time() - debut
                self.evolution.append((duree,s.getValeur()))
        return record
    
    def multistart_LS_Swap (self, nb_iter = 20):
        record = Solution(self.instance, 'Multistart_LS_Swap')
        debut = time.time()
        self.evolution = []
        for iter in range(nb_iter):
            s = self.compute_random()
            self.swap(s)
            if record.getValeur() == 0.0 or s.getValeur() < record.getValeur():
                record.setSequence(s.getSequence())
                duree = time.time() - debut
                self.evolution.append((duree,s.getValeur()))
        return record

    def multistart_LS_OrOpt (self, nb_iter = 20):
        record = Solution(self.instance, 'Multistart_LS_OrOpt')
        debut = time.time()
        self.evolution = []
        for iter in range(nb_iter):
            s = self.compute_random()
            self.OrOpt(s)
            if record.getValeur() == 0.0 or s.getValeur() < record.getValeur():
                record.setSequence(s.getSequence())
                duree = time.time() - debut
                self.evolution.append((duree,s.getValeur()))
        return record

    def regen(self, s: Solution):
        # can be used for recherche_tabou or simulated annealing
        seq = s.getSequence()
        
        fst = random.randint(0, len(seq)-1)
        snd = random.randint(0, len(seq)-1)
        seq[fst], seq[snd] = seq[snd], seq[fst]
        
        s.setSequence(seq)
        return s

    def hash(self, s: Solution)->str:
        hashstr = ""
        seq = s.getSequence()
        for id in seq: 
            hashstr += str(id)

        return hashstr
    
    def recherche_tabou(self, nb_iter = 20): 
        record = Solution(self.instance, 'Recherche Tabou')
        debut = time.time()
        self.evolution = []
        
        forbiden_hashes:list[str] = []
        s = self.compute_random()

        for iter in range(nb_iter):
            self.regen(s)
            shash = self.hash(s)
            
            if shash in forbiden_hashes:
                continue

            if record.getValeur() == 0.0 or s.getValeur() < record.getValeur():
                record.setSequence(s.getSequence())
                duree = time.time() - debut
                self.evolution.append((duree,s.getValeur()))
            else: 
                forbiden_hashes.append(shash)
            
        return record



if __name__ == '__main__':

    place = "Le Havre, Seine-Maritime, France"
    amenity = "charging_station"
    tags = {"amenity": amenity}

    # 1. Download the walkable street network (for background)
    G = ox.graph_from_place(place, network_type="walk")
    G_proj = ox.project_graph(G)

    # 2. Download restaurant POIs
    restaurants = ox.features_from_place(place, tags)
    restaurants_proj = restaurants.to_crs(G_proj.graph['crs'])
    print(f"found {len(restaurants)} POIs")

    # 3. Plot street network (light grey) + restaurants (red dots)
    fig, ax = ox.plot_graph(
        G_proj,
        show=False,
        close=False,
        bgcolor='white',
        node_color='lightgrey',
        edge_color='lightgrey',
        node_size=5,
        edge_linewidth=0.5
    )

    # Plot restaurants on top
    restaurants_proj.plot(ax=ax, color='red', markersize=10, alpha=0.8, label='Restaurants')

    # Final touches
    ax.set_title("Restaurants in Le Havre", fontsize=14)
  
    plt.show()


    """
    # Define place and tag
    place = "Le Havre, Seine-Maritime, France"
    tags = {"amenity": "restaurant"}

    # 1. Download only restaurant POIs
    restaurants = ox.features_from_place(place, tags)

    # 2. Print info
    print(f"Found {len(restaurants)} restaurants")
    print(restaurants[["name", "geometry"]].head())

    # 3. Plot restaurants directly
    fig, ax = plt.subplots()
    restaurants.plot(ax=ax, color='red', markersize=10, alpha=0.7)
    ax.set_title("Restaurants in Le Havre")
    plt.show()

    # 4. Save to GeoJSON
    restaurants.to_file("./data/restaurants_lehavre.geojson", driver="GeoJSON")
    
    
    G = ox.graph_from_place(place, simplify=True, retain_all=False, custom_filter="")
    G_proj = ox.project_graph(G)
    fig, ax = ox.plot_graph(G_proj, show=True, close=False)
    plt.show()
    """




    """
    filepath = "./data/lehavre-restaurants.graphml"
    place = "Le Havre, Seine-Maritime, France"

    # 1. Load walkable street network
    G = ox.graph_from_place(place, network_type="walk")
    G_proj = ox.project_graph(G)

    # 2. Load restaurant POIs
    tags = {"amenity": "restaurant"}
    restaurants = ox.features_from_place(place, tags)

    # 3. Project restaurant GeoDataFrame to match graph CRS
    restaurants_proj = restaurants.to_crs(G_proj.graph['crs'])

    # 4. Plot the graph
    fig, ax = ox.plot_graph(G_proj, show=False, close=False)

    # 5. Plot the restaurants
    restaurants_proj.plot(ax=ax, color='red', markersize=10, alpha=0.7, label='Restaurants')

    
    ax.set_title("Restaurants in Le Havre")
    plt.show()

    # 7. Save restaurant POIs to GeoJSON
    restaurants.to_file("restaurants_lehavre.geojson", driver='GeoJSON')
"""

   
    """
    place = "Le Havre, Seine-Maritime, France"
   
    # Download Open Street Map data.
    tags = {"amenity": "restaurant"}
    restaurants = ox.features_from_place(place, tags)
   

    # Print number of restaurants and preview
    print(f"Found {len(restaurants)} restaurants")
    print(restaurants[["name", "geometry"]].head())


    # Project graph and restaurants to same CRS
    G_proj = ox.project_graph(restaurants)
    restaurants_proj = restaurants.to_crs(G_proj.graph['crs'])

    # Plot the graph
    fig, ax = ox.plot_graph(G_proj, show=False, close=False)

    # Plot the restaurants
    restaurants_proj.plot(ax=ax, color='red', markersize=10, alpha=0.7, label='Restaurants')

    # Add a legend and title
    ax.set_title("Restaurants in Le Havre")
    plt.show()

    restaurants.to_file("restaurants_lehavre.geojson", driver='GeoJSON')

    #fig, ax = ox.plot.plot_graph(G, show=True, save=True, close=True, filepath="./images/lh.svg")

    # get all building footprints and save as a geopackage via geopandas
   
"""
    """
    random.seed(0)
    filename = "./data/instance3.txt"
    print(f"[metrics._main_]: About to load instance from file {filename}.")
    inst = Instance.fromFile(filename)
    print(f"[metrics._main_]: Loaded instance # nodes = {inst.nb_sommets}.")
    print(f"[metrics._main_]: About to reduce loaded graph to Hubs only graph (redhog).")

    proportion = 15
    print(f"[metrics._main_]: Hub to Node radius set to {proportion}% of delta(mindist, maxdist).")
    hoginst = inst.redhog(proportion)
  

    # generation heuristique des solutions
    heur = Heuristiques(hoginst)
 
    
    methodes = [heur.compute_triviale, heur.compute_random, heur.compute_nearest,heur.ilp, heur.multistart, heur.multistart_LS_2Opt, heur.multistart_LS_Swap, heur.multistart_LS_OrOpt, heur.recherche_tabou]
    for m in methodes:
        print(f"\n[metrics._main_]: About to run the {m.__name__} strategy.")
        debut = time.time()
        sol:Solution = m()
        duree = time.time() - debut
        sol.setTemps(duree)
        sol.affiche()
        sol.plot()
        print('evolution = ', heur.evolution)
        if len(heur.evolution) > 0:
            heur.plot_evo()
    """