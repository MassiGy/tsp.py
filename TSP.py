# -*- coding: utf-8 -*-

import random
import math
import os
import itertools
import time
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD
from pathlib import Path
import osmnx as ox
import geopandas as gpd
from shapely import LineString
from shapely.geometry import Point


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
                
                if self.maxdist < dist and siid != sjid:
                    self.maxdist = dist
            
                if self.mindist > dist and siid != sjid:
                    self.mindist = dist


    # reduce to hubs only graph
    def redhog(self, hubradius, debugplot=True):
        print(f"[metrics.redhog]: Reduce graph to hubs only graph, input graph # nodes={self.nb_sommets}")
        print(f"[metrics.redhog]: maxdist={self.maxdist}, mindist={self.mindist}")
        
        toHubDist = mapval(hubradius, 0, 100, self.mindist, self.maxdist)
        print(f"[metrics.redhog]: Node To Hub Distance Radius={toHubDist}")
        
        hubdeg = 2
        print(f"[metrics.redhog]: Hub Min Degree={hubdeg}")
        
        outliers: list[Sommet] = []
        n2ns: dict[Sommet, list[Sommet]] = {}

        for si in self.sommets:
            n2ns[si] = []
            for sj in self.sommets:
                if si.getId() == sj.getId():
                    continue
                if self.dist[si.getId()][sj.getId()] <= toHubDist:
                    n2ns[si].append(sj)

        candidates = sorted(n2ns.items(), key=lambda item: -len(item[1]))
        hubs: list[Sommet] = []
        for n, neighbors in candidates:
            if len(neighbors) >= hubdeg and n not in hubs:
                hubs.append(n)

        if debugplot:
            Instance.visualize_hubs(n2ns, hubs, name="(All potential hubs)")

        # Iteratively resolve overlaps via cherry-picking
        change = True
        while change:
            change = False
            new_n2ns = {hub: set(n2ns[hub]) for hub in hubs}
            hubs_to_remove = set()

            for i in range(len(hubs)):
                hi = hubs[i]
                hi_id = hi.getId()
                for j in range(i + 1, len(hubs)):
                    hj = hubs[j]
                    hj_id = hj.getId()

                    if self.dist[hi_id][hj_id] > toHubDist:
                        continue

                    overlap = set(n2ns[hi]).intersection(n2ns[hj])
                    if not overlap:
                        continue

                    # Cherry-pick overlapping nodes
                    for node in overlap:
                        # Assign to the nearest hub
                        dist_to_hi = self.dist[hi_id][node.getId()]
                        dist_to_hj = self.dist[hj_id][node.getId()]
                        if dist_to_hi <= dist_to_hj:
                            new_n2ns[hi].add(node)
                            if node in new_n2ns[hj]:
                                new_n2ns[hj].remove(node)
                        else:
                            new_n2ns[hj].add(node)
                            if node in new_n2ns[hi]:
                                new_n2ns[hi].remove(node)

                        # If overlapping node is a hub itself — merge it
                        if node in hubs:
                            hubs_to_remove.add(node)
                            new_n2ns[hi if dist_to_hi <= dist_to_hj else hj].update(n2ns[node])

                            # Also merge its connections
                            del new_n2ns[node]

                    change = True

            # Remove merged hubs
            hubs = [h for h in hubs if h not in hubs_to_remove]
            n2ns = {hub: list(neighs) for hub, neighs in new_n2ns.items()}

        if debugplot:
            Instance.visualize_hubs(n2ns, hubs, name="(Reduced hubs set - overlap resolved)")

        print(f"[metrics.redhog]: # hubs in input graph={len(hubs)}")

        hubsClustersNodes: list[Sommet] = hubs.copy()
        for n in hubs:
            hubsClustersNodes.extend(n2ns[n])

        hubsClustersNodes = set(hubsClustersNodes)
        outliers = list(set(self.sommets) - hubsClustersNodes)
        print(f"[metrics.redhog]: # outliers in input graph={len(outliers)}. About to add them to the reduced graph nodes list.")
        hubs.extend(outliers)

        for i, n in enumerate(hubs):
            n.setId(i)

        if debugplot:
            Instance.visualize_hubs(n2ns, hubs, name="(Hubs+Outliers - Final reduced Graph)")

        hoginst = Instance(str("redhog " + self.name), len(hubs), hubs)
        hoginst.computeDistances()
        print(f"[metrics.redhog]: Reduced graph # nodes={len(hubs)}")

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
       
        # dynamic boundries
        for i in range(self.nb_sommets):
            s = self.sommets[i]
            sx, sy = s.getX(), s.getY()
            
            if i == 0: 
                plot_xlim = sx
                plot_ylim = sy
                plot_xmin = sx
                plot_ymin = sy
                continue

            if sx > plot_xlim:
                plot_xlim = sx+1
            if sy > plot_ylim:
                plot_ylim = sy+1
            
            if sx < plot_xmin:
                plot_xmin = sx - 1 
            if sy < plot_ymin:
                plot_ymin = sy - 1

        plt.figure()
        plt.title('instance {}: {} sommets'.format(self.name, self.nb_sommets))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(plot_xmin,plot_xlim)
        plt.ylim(plot_ymin,plot_ylim)
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

    def csv_str(self) -> str:
        # this is useful for our plot_results.py script
        return "{},{:.6f},{:.6f}".format(self.name, self.valeur, self.temps)

    def affiche (self):
        print('solution \'{}\': {} -> val = {:.6f} temps = {:.6f} s'.format(self.name, self.sequence, self.valeur, self.temps))

    def plot (self):
        # dynamic boundries for the plot
        for i in range(self.instance.nb_sommets):
            s = self.instance.sommets[i]
            sx, sy = s.getX(), s.getY()
            if i == 0: 
                plot_xlim = sx
                plot_ylim = sy
                plot_xmin = sx
                plot_ymin = sy
                continue

            if sx > plot_xlim:
                plot_xlim = sx+1
            if sy > plot_ylim:
                plot_ylim = sy+1
            
            if sx < plot_xmin:
                plot_xmin = sx - 1 
            if sy < plot_ymin:
                plot_ymin = sy - 1
        
        plt.figure()
        plt.title('\'{}\': valeur = {:.2f} en {:.2f} s'.format(self.name, self.valeur, self.temps))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(plot_xmin,plot_xlim)
        plt.ylim(plot_ymin,plot_ylim)
        
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
        plt.xlabel('time (s)')
        plt.ylabel('distsum (km)')
        x = [elt[0] for elt in self.evolution]
        y = [elt[1]/1000 for elt in self.evolution]
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
        """
        Changes two arcs and check if the value is better. If so, update 
        the current sequent to reflect that change and return.
        """
        seq = s.getSequence()
        
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
                    return True
        return False
    
    def OrOpt (self, s: Solution, nb_iter=20):
        """
        Changes a node position randomly and check if the value is better. If so, update 
        the current sequent to reflect that change and return.
        """
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
                return True
        
            counter +=1

        return False

    def swap (self, s: Solution, nb_iter=20):
        """
        Changes two nodes randomly and check if the value is better. If so, update 
        the current sequent to reflect that change and return.
        """
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
                return True
        
            counter +=1

        return False

    def multistart (self, nb_iter = 20):
        record = Solution(self.instance, 'Multistart_Without_LS')
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

        for iter in range(nb_iter):
            s = self.compute_random()
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
    use_osmnx_data = False
    
    
    if not use_osmnx_data:
        random.seed(0)
        filename = "./data/instance2.txt"
        print(f"[metrics._main_]: Set to not use osmnx data. Use localy generated data instead.")
        print(f"[metrics._main_]: About to load instance from file {filename}.")
        inst = Instance.fromFile(filename)
        print(f"[metrics._main_]: Loaded instance # nodes = {inst.nb_sommets}.")
        print(f"[metrics._main_]: About to reduce loaded graph to Hubs only graph (redhog).")

        proportion = 15
        print(f"[metrics._main_]: Hub to Node radius set to {proportion}% of delta(mindist, maxdist).")
        hoginst = inst.redhog(proportion)
    

        # generation heuristique des solutions
        heur = Heuristiques(hoginst)
    
        # wave1: with no nb_iter arg and no evolution tracking
        methodes = [heur.compute_triviale, heur.compute_random, heur.compute_nearest,heur.ilp]
        for m in methodes:
            print(f"\n[metrics._main_]: About to run the {m.__name__} strategy.")
            debut = time.time()
            sol:Solution = m()
            duree = time.time() - debut
            sol.setTemps(duree)
            sol.affiche()
            sol.plot()

        # wave2: with nb_iter arg and evolution tracking
        methodes = [heur.multistart, heur.multistart_LS_2Opt, heur.multistart_LS_Swap, heur.multistart_LS_OrOpt, heur.recherche_tabou]
        for m in methodes:
            print(f"\n[metrics._main_]: About to run the {m.__name__} strategy.")
            debut = time.time()
            sol:Solution = m(nb_iter=inst.nb_sommets*inst.nb_sommets)
            duree = time.time() - debut
            sol.setTemps(duree)
            sol.affiche()
            sol.plot()
            print('evolution = ', heur.evolution)
            if len(heur.evolution) > 0:
                heur.plot_evo()


    else:
        # osmnx data target
        place = "Le Havre, Seine-Maritime, France"
        amenity = "charging_station"
        print(f"[metrics._main_]: Set to use osmnx data w/ place={place} and amenity={amenity}.")
       
        # local cache
        graph_cache_filepath = "./data/lh.graphml"
        print(f"[metrics._main_]: Local graph cache filepath={graph_cache_filepath} .")
        
        amenity_cache_storage_driver ="GeoJSON"
        amenity_cache_filepath = f"./data/lh_amenities.{amenity}.{amenity_cache_storage_driver.lower()}"
        print(f"[metrics._main_]: Local amenities cache filepath={amenity_cache_filepath} .")
        
        # Load or download the graph
        G=None
        if os.path.exists(graph_cache_filepath) and os.path.getsize(graph_cache_filepath) > 0:
            print("[metrics._main_]: Loading graph from cache...")
            G = ox.load_graphml(graph_cache_filepath)
        else:
            print("[metrics._main_]: Downloading graph...")
            G = ox.graph_from_place(place, network_type="walk")
            ox.save_graphml(G, graph_cache_filepath)

        # Project the graph
        G_proj = ox.project_graph(G)

        # Load or download POIs
        POIs=None
        if os.path.exists(amenity_cache_filepath) and os.path.getsize(amenity_cache_filepath) > 0:
            print("[metrics._main_]: Loading amenities from cache...")
            POIs = gpd.read_file(amenity_cache_filepath)
        else:
            print("[metrics._main_]: Downloading amenities...")
            tags = {"amenity": amenity}
            POIs = ox.features_from_place(place, tags)
            POIs.to_file(amenity_cache_filepath, driver=amenity_cache_storage_driver)




        POIs_proj = POIs.to_crs(G_proj.graph['crs'])
        lenPOIs = len(POIs)
        print(f"[metrics._main_]: Got {lenPOIs} POIs")

        # Plot street network (light grey) + POIs (red dots)
        print("[metrics._main_]: Ploting map with all POIs...")
        
        fig, ax = ox.plot_graph(
            G_proj, show=False, close=False,
            bgcolor='white', node_color='lightgrey', edge_color='lightgrey',
            node_size=5, edge_linewidth=0.5
        )

        # Plot POIs on top
        POIs_proj.plot(ax=ax, color='red', markersize=10, alpha=0.8, label=amenity)
        
        ax.set_title(f"Le Havre's {amenity} network", fontsize=14)
        plt.show()
    
        


        # Convert the amenities coordinates to a list of points.
        poi_xy_tuples = []

        for geom in POIs_proj.geometry:
            if isinstance(geom, Point):
                poi_xy_tuples.append((geom.x, geom.y))
            
            elif geom is not None and geom.is_valid:
                # Fallback: use centroid of the geometry
                centroid = geom.centroid
                poi_xy_tuples.append((centroid.x, centroid.y))


        nodes = [Sommet(p[0], p[1]) for p in poi_xy_tuples]
        inst = Instance(f"Le Havre's {amenity} network", len(poi_xy_tuples), nodes)
        inst.computeDistances()
        
        proportion = 12
        print(f"[metrics._main_]: Hub to Node radius set to {proportion}% of delta(mindist, maxdist).")
        hoginst = inst.redhog(proportion, debugplot=False)
    
        # Plot street network (light grey) + reduced set of POIs (red dots)
        print("[metrics._main_]: Ploting map with the reduced set of POIs (after redhog routine)...")
       
        fig, ax = ox.plot_graph(
            G_proj, show=False, close=False,
            bgcolor='white', node_color='lightgrey', edge_color='lightgrey',
            node_size=5, edge_linewidth=0.5
        )

        gdf4redhog = gpd.GeoDataFrame(geometry=[Point(s.getX(), s.getY()) for s in hoginst.sommets], crs=G_proj.graph['crs'])
        gdf4redhog.plot(ax=ax, color='red', markersize=10, alpha=0.8, label=amenity)

        ax.set_title(f"Le Havre's {amenity} network (redhoged)", fontsize=14)
        plt.show()
        



        # generation heuristique des solutions
        heur = Heuristiques(hoginst)
    
        
        # wave1: with no nb_iter arg and no evolution tracking
        methodes = [heur.compute_triviale, heur.compute_random, heur.compute_nearest, heur.ilp]
        for m in methodes:
            print(f"\n[metrics._main_]: About to run the {m.__name__} strategy.")
            debut = time.time()
            sol:Solution = m()
            duree = time.time() - debut
            sol.setTemps(duree)
            print(sol.csv_str())
            
            # plot the graph
            print("[metrics._main_]: Ploting results on to the map...")
            
            fig, ax = ox.plot_graph(
            G_proj, show=False, close=False,
            bgcolor='white', node_color='lightgrey', edge_color='lightgrey',
            node_size=5, edge_linewidth=0.5
            )

            # plot the points
            gdf4redhog = gpd.GeoDataFrame(geometry=[Point(s.getX(), s.getY()) for s in hoginst.sommets], crs=G_proj.graph['crs'])
            gdf4redhog.plot(ax=ax, color='red', markersize=10, alpha=0.8, label=amenity)

            # plot the edges (lines between the points of the solution sequence)
            seq = sol.getSequence()
            seq.append(seq[0])
            lines = [
                LineString([
                    (hoginst.sommets[seq[i]].getX(), hoginst.sommets[seq[i]].getY()), 
                    (hoginst.sommets[seq[i+1]].getX(), hoginst.sommets[seq[i+1]].getY())
                ]) 
                for i in range(len(seq) - 1)
            ]
            gdf_lines = gpd.GeoDataFrame(geometry=lines, crs=G_proj.graph['crs'])
            gdf_lines.plot(ax=ax, color='blue', linewidth=1, linestyle='--', alpha=0.7, label='Edges')


            # set title and show the plot
            ax.set_title(f"TSP on Le Havre's {amenity} network\nstrategy={m.__name__}, distsum={"{:.3f}".format(sol.getValeur()/1000)} km, time={"{:.6f}".format(duree)} seconds", fontsize=10)
            plt.show()
            

        # wave2: with nb_iter arg and evolution tracking
        methodes = [heur.multistart, heur.multistart_LS_2Opt, heur.multistart_LS_Swap, heur.multistart_LS_OrOpt, heur.recherche_tabou]
        for m in methodes:
            print(f"\n[metrics._main_]: About to run the {m.__name__} strategy.")
            debut = time.time()
            sol:Solution = m(nb_iter=lenPOIs*lenPOIs)
            duree = time.time() - debut
            sol.setTemps(duree)
            print(sol.csv_str())
        
            # plot the graph
            print("[metrics._main_]: Ploting results on to the map...")
            
            fig, ax = ox.plot_graph(
            G_proj, show=False, close=False,
            bgcolor='white', node_color='lightgrey', edge_color='lightgrey',
            node_size=5, edge_linewidth=0.5
            )

            # plot the points
            gdf4redhog = gpd.GeoDataFrame(geometry=[Point(s.getX(), s.getY()) for s in hoginst.sommets], crs=G_proj.graph['crs'])
            gdf4redhog.plot(ax=ax, color='red', markersize=10, alpha=0.8, label=amenity)

            # plot the edges (lines between the points of the solution sequence)
            seq = sol.getSequence()
            seq.append(seq[0])
            lines = [
                LineString([
                    (hoginst.sommets[seq[i]].getX(), hoginst.sommets[seq[i]].getY()), 
                    (hoginst.sommets[seq[i+1]].getX(), hoginst.sommets[seq[i+1]].getY())
                ]) 
                for i in range(len(seq) - 1)
            ]
            gdf_lines = gpd.GeoDataFrame(geometry=lines, crs=G_proj.graph['crs'])
            gdf_lines.plot(ax=ax, color='blue', linewidth=1, linestyle='--', alpha=0.7, label='Edges')


            # set title and show the plot
            ax.set_title(f"TSP on Le Havre's {amenity} network\nstrategy={m.__name__}, distsum={"{:.3f}".format(sol.getValeur()/1000)} km, time={"{:.6f}".format(duree)} seconds", fontsize=10)
            plt.show()
            


            print('evolution = ', heur.evolution)
            if len(heur.evolution) > 0:
                heur.plot_evo()
                continue
