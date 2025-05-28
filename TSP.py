# -*- coding: utf-8 -*-

import random
import math
import itertools
import time
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD


class Sommet:
    cpt_sommet = 0
    
    def __init__ (self, x, y):
        self.id = Sommet.cpt_sommet
        Sommet.cpt_sommet += 1
        
        self.x = x
        self.y = y
    
    def getId (self):
        return self.id
    
    def getX (self):
        return self.x
    
    def getY (self):
        return self.y
    
    def affiche (self):
        print('({}:{:.2f},{:.2f})'.format(self.id,self.x,self.y))


class Instance:
    def __init__ (self, name, n):
        self.name = name
        self.nb_sommets = n
        self.reset()
    
    def size (self):
        return self.nb_sommets
    
    def reset (self):
        self.generateNodes()
        self.computeDistances()

    def generateNodes (self):
        self.sommets = [Sommet(random.uniform(0,100), random.uniform(0,100)) for i in range(self.nb_sommets)]
    
    def computeDistances (self):
        self.dist = [[0.0] * self.nb_sommets for i in range(self.nb_sommets)]
        for si in self.sommets:
            for sj in self.sommets:
                delta_x = si.getX() - sj.getX()
                delta_y = si.getY() - sj.getY()
                self.dist[si.getId()][sj.getId()] = math.sqrt(delta_x ** 2 + delta_y ** 2)
    
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
        print('solution \'{}\': {} -> val = {:.2f} temps = {:.2f} s'.format(self.name, self.sequence, self.valeur, self.temps))

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


    def mvt2Opt (self, s: Solution):
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
    inst = Instance('random',30)
    # inst.affiche()
    inst.plot()
    
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
    
    methodes = [heur.ilp,heur.compute_triviale, heur.compute_random, heur.compute_nearest, heur.multistart, heur.multistart_LS]
    for m in methodes:
        debut = time.time()
        sol = m()
        duree = time.time() - debut
        sol.setTemps(duree)
        sol.affiche()
        sol.plot()
        print('evolution = ', heur.evolution)
        if len(heur.evolution) > 0:
            heur.plot()
