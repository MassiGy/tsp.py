# -*- coding: utf-8 -*-

import random
import math
import itertools
import time
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD


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
            s.affiche()
        

        instance.computeDistances()
        return instance
    
    def computeDistances (self):
        self.maxdist = -float('inf')
        self.mindist = float('inf')
 
        self.dist = [[0.0] * self.nb_sommets for i in range(self.nb_sommets)]
        for si in self.sommets:
            for sj in self.sommets:
                delta_x = si.getX() - sj.getX()
                delta_y = si.getY() - sj.getY()
                self.dist[si.getId()][sj.getId()] = math.sqrt(delta_x ** 2 + delta_y ** 2)

                dist = self.dist[si.getId()][sj.getId()]
                
                # this is the init
                if self.maxdist == -1.0 and si.getId() != sj.getId():
                    self.maxdist = dist
                
                if self.mindist == -1.0 and si.getId() != sj.getId():
                    self.mindist = dist

                if self.maxdist < dist and si.getId() != sj.getId():
                    self.maxdist = dist
            
                if self.mindist > dist and si.getId() != sj.getId():
                    self.mindist = dist


    # reduce to hubs only graph
    def redhog(self, hubradius):
        print(f"maxdist={self.maxdist}, mindist={self.mindist}\n")
        toHubDist = mapval(hubradius, 0, 100, self.mindist, self.maxdist)
        
        hubs:list[Sommet] = []
        hubdeg = 4
       
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
                
     
        for n in n2ns:
            if len(n2ns[n]) >= hubdeg:
                # here we consider n as a hub
                hubs.append(n)

                # make sure that n is no longer reachable
                # by other hubs
                for _n in n2ns:
                    if _n.getId() != n.getId() and n in n2ns[_n]:
                        n2ns[_n].remove(n)

                
        Instance.visualize_hubs(n2ns, hubs)
        
        # reduce the overlaps 
        to_remove = set()
        for si in hubs:
            for sj in hubs:
                siid, sjid= si.getId(),sj.getId()
                
                if siid == sjid:
                    continue

                l1 = len(n2ns[si])
                l2 = len(n2ns[sj])

                if sj in n2ns[si] or si in n2ns[sj]: 
                    if l1 > l2:
                        n2ns[si] = list(set(n2ns[si]) | set(n2ns[sj]))
                        to_remove.add(sj)
                    else: 
                        n2ns[sj] = list(set(n2ns[sj]) | set(n2ns[si]))
                        to_remove.add(si)

                    if si in n2ns[si]: n2ns[si].remove(si)
                    if sj in n2ns[sj]: n2ns[sj].remove(sj)

                else:
                    overlap = list(set(n2ns[si]) & set(n2ns[sj]))
                    l3 = len(overlap)

                    for s in overlap:
                        sid = s.getId()
                        if self.dist[siid][sid] <= self.dist[sjid][sid]:
                            if l2-1 >= hubdeg-1: 
                                n2ns[sj].remove(s)
                                l2 -=1
                        else:
                            if l1 -1 >= hubdeg-1:
                                n2ns[si].remove(s)
                                l1 -=1
        

        hubs = [h for h in hubs if h not in to_remove]
        Instance.visualize_hubs(n2ns, hubs)     
       

        hubsClustersNodes:list[Sommet] = hubs.copy()
        for n in hubs:
            hubsClustersNodes.extend(n2ns[n])
        
        hubsClustersNodes = set(hubsClustersNodes)

        outliers = list(set(self.sommets) - set(hubsClustersNodes))
        hubs.extend(outliers)


        


        for i in range(len(hubs)):
            # update the ids for the hubs
            n = hubs[i]
            n.setId(i)


        Instance.visualize_hubs(n2ns, hubs)     
            

        

       
        # now that we have the hubs, create a new instance
        # using those hubs as nodes
        hoginst = Instance(str("redhog "+self.name), len(hubs), hubs)
        hoginst.computeDistances()


        return hoginst
    

    
    @classmethod
    def visualize_hubs(cls, n2ns: dict, hubs: list):
        plt.figure(figsize=(10, 10))
        
        # Plot all nodes
        for node, neighbors in n2ns.items():
            plt.scatter(node.x, node.y, color='grey', s=30, zorder=1)
        
        # Highlight hubs and their connections
        for hub in hubs:
            plt.scatter(hub.x, hub.y, color='red', s=100, zorder=3)
            
            for connected_node in n2ns[hub]:
                plt.plot([hub.x, connected_node.x], [hub.y, connected_node.y], 
                        color='blue', linewidth=1, zorder=2)

        plt.title("Hub-Node Connections")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis("equal")
        plt.show()

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
        plt.show()


class Solution:
    def __init__ (self, inst, desc):
        self.instance = inst
        self.name = desc
        self.sequence = [-1] * inst.size()
        self.valeur = 0.0
        self.temps = 0.0

    def getSequence (self):
        return self.sequence

    def getValeur (self):
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

    def plot (self):
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
            dist = 200.0
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

    def ilp(self):  # programmation linéaire en nombre entier. PLNE
        record = Solution(self.instance, 'programmation linéaire en nombre entier. PLNE')

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


    # TODO: update this method to use the same logic as the others,
    #  meaning: 
    #  * start by initializing a record:Solution
    #  * do your logic,
    #  * save the result in the record 
    #  * return the record.

    # But, do we really need to change that since it is used 
    # in the localsearch function and not directly as a hurestic
    # it is more of a helper method then a hurestic
    
    # it is like the regen method in the simulated annealing algo
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
    
    def localSearch (self, s):
        cpt = 0
        while self.mvt2Opt(s) is True:
            # print('iteration {}'.format(cpt),end='')
            # s.affiche()
            cpt += 1
        return cpt
    
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
    
    def multistart_LS (self, nb_iter = 20):
        record = Solution(self.instance, 'Multistart_LS')
        debut = time.time()
        self.evolution = []
        for iter in range(nb_iter):
            s = self.compute_random()
            self.localSearch(s)
            if record.getValeur() == 0.0 or s.getValeur() < record.getValeur():
                record.setSequence(s.getSequence())
                duree = time.time() - debut
                self.evolution.append((duree,s.getValeur()))
        return record
    




if __name__ == '__main__':
    # creation de l'instance: 10 sommets
    random.seed(0)
    inst = Instance.fromFile("./data/instance2.txt")
    inst.plot()
    inst.redhog(20).plot()
    
    # generation heuristique des solutions
    heur = Heuristiques(inst)
 
    
    # debut = time.time()
    # s1 = heur.compute_triviale()
    # duree = time.time() - debut
    # print('heuristique triviale: duree = {:.3f} s'.format(duree))
    # s1.affiche()
    
    # debut = time.time()
    # s2 = heur.compute_random()
    # duree = time.time() - debut
    # print('heuristique random: duree = {:.3f} s'.format(duree))
    # s2.affiche()
    
    # debut = time.time()
    # s3 = heur.compute_nearest()
    # duree = time.time() - debut
    # print('heuristique plus proche voisins: duree = {:.3f} s'.format(duree))
    # s3.affiche()
    # s3.plot()
    
    # debut = time.time()
    # heur.localSearch(s3)
    # duree = time.time() - debut
    # print('recherche locale: duree = {:.3f} s'.format(duree))
    # s3.affiche()
    # s3.plot()
    
    # debut = time.time()
    # s4, evolution = heur.multistart(20)
    # duree = time.time() - debut
    # print('multistart: duree = {:.3f} s'.format(duree))
    # s4.affiche()
    # s4.plot()
    # print('evolution = ', evolution)
    
    # debut = time.time()
    # s5 = heur.compute_enumerate()
    # duree = time.time() - debut
    # print('multistart: duree = {:.3f} s'.format(duree))
    # s5.affiche()
    # s5.plot()
    """
    methodes = [heur.compute_triviale, heur.compute_random, heur.compute_nearest, heur.mvt2Opt, heur.multistart, heur.multistart_LS]
    for m in methodes:
        debut = time.time()
        sol:Solution = m()
        duree = time.time() - debut
        sol.setTemps(duree)
        sol.affiche()
        sol.plot()
        print('evolution = ', heur.evolution)
        if len(heur.evolution) > 0:
            heur.plot()
    """
