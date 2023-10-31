
from scipy.spatial import distance
from operator import itemgetter
from hashlib import sha256
import numpy as np
import time
import math
from collections import Counter
import heapq
import treap
import copy
from itertools import product
from multiprocessing import Pool, Lock

class Clustering:
    def __init__(self, S, M, typeproblem, limit_cluster_size_percent, limit_cluster_volume_percent, distance_method):
        self.S = S
        self.M = M
        self.distance = []
        self.position = []
        self.step = 0
        self.cluster = set()
        self.cluster_iteration = []
        self.rows_del = set()
        self.row_del_queue = { i : False for i in range(len(self.S))}
        self.total_cluster = len(self.S)
        self.priorityqueue = []
        self.priorityqueue_treap = treap.treap()

        if distance_method == 'hamming':
            self.hamming_cache = {bytearray(e).decode("utf-8") : e for e in self.S}

        start = time.time()
        self.recalculate_distance(distance_method)
        end = time.time()
        print(f"[Calculate distance] Duración (segundo): {end - start}")

        self.total_volume = np.linalg.norm(self.S) * S.shape[0] * S.shape[1]

        start = time.time()
        while self.merge_cluster(limit_cluster_size_percent, limit_cluster_volume_percent): 
            pass
            
        end = time.time()
        print(f"[Agglomerative] Duración (segundo): {end - start}")
        print("complete!")

    def process_distances(self, data):
        i, j, distance_method, lock = data
        
        if distance_method == "euclidean":
            self.M[i, j] = distance.euclidean(self.S[i], self.S[j])
        elif distance_method == "manhattan":
            self.M[i, j] = distance.cityblock(self.S[i], self.S[j])
        elif distance_method == "hamming":
        
            _i = bytearray(self.S[i]).decode("utf-8") 
            _j = bytearray(self.S[j]).decode("utf-8") 
            self.M[i, j] =  round(distance.hamming(self.hamming_cache[_i], self.hamming_cache[_j]) * len(self.S[i]))
        with lock: 
            heapq.heappush(self.priorityqueue, (self.M[i, j],  (i, j)))
            return self.M[i, j]

    def recalculate_distance_parallel(self, distance_method):
        sum_dist = 0.0
        print("Size matrix: ", len(self.S))
        euclidean_cache = {}
        lock = Lock()
        with Pool(10) as pool:
            for result in pool.starmap(self.process_distances, product(range(len(self.S)), range(len(self.S)), [distance_method], [lock])):
                sum_dist += result
                
        if distance_method == 'hamming':
            self.hamming_cache.clear()
        
        self.priorityqueue = sorted(self.priorityqueue)
        print(self.priorityqueue[:10])
        print("distance: ", distance_method)
        print("sum_dist: ", sum_dist)

    def recalculate_distance(self, distance_method):
        sum_dist = 0.0
        print("size matrix: ", len(self.S))
        euclidean_cache = {}
        for i in range(len(self.S)):
            for j in range(len(self.S)):  
                if i <= j:
                    continue
                if distance_method == "euclidean":
                    self.M[i, j] = distance.euclidean(self.S[i], self.S[j])
                elif distance_method == "manhattan":
                    self.M[i, j] = distance.cityblock(self.S[i], self.S[j])
                elif distance_method == "hamming":
                
                    _i = bytearray(self.S[i]).decode("utf-8") 
                    _j = bytearray(self.S[j]).decode("utf-8") 
                    self.M[i, j] =  round(distance.hamming(self.hamming_cache[_i], self.hamming_cache[_j]) * len(self.S[i]))
                 
                heapq.heappush(self.priorityqueue, (self.M[i, j],  (i, j)))
                sum_dist += self.M[i,j]
            print(f"complete (processing distances): { i * 100 / len(self.S):.2f} %", end='\r')

        if distance_method == 'hamming':
            self.hamming_cache.clear()
        self.priorityqueue = sorted(self.priorityqueue)
        
        print("distance: ", distance_method)
        print("sum_dist: ", sum_dist)
     
    def volumen(self, solution, new_solution):
        if self.cluster_iteration:

            cluster_a = set()
            cluster_b = set()
            for c in self.cluster_iteration[-1]:
                if (not cluster_a) and solution in c:
                    cluster_a = c

                if (not cluster_b) and new_solution in c:
                    cluster_b = c
                
                if cluster_a and cluster_b:
                    break

            cluster_merge = cluster_a.union(cluster_b)
            l, w = len(cluster_merge), self.S.shape[1]
            return (np.linalg.norm(itemgetter(*cluster_merge)(self.S)) * l * w) * 100 / self.total_volume
        else:
            return 0
    
    def limit_size_cluster(self, solution, new_solution):
        if self.cluster_iteration:
            cluster_a = set()
            cluster_b = set()
            
            for c in self.cluster_iteration[-1]:
                if (not cluster_a) and solution in c:
                    cluster_a = c

                if (not cluster_b) and new_solution in c:
                    cluster_b = c
                
                if cluster_a and cluster_b:
                    break

            cluster_merge = cluster_a.union(cluster_b)
            return len(cluster_merge) * 100 / len(self.S) 
        else:
            return 0
    
    def merge_minimal_distance(self, limit_size_percen, volumen_percen):
        d = None
        p = None
        min_distance = float('inf')
        for i in range(len(self.M)):
            if i in self.rows_del:
                continue
            for j in range(len(self.M)):
                if i <= j:
                    continue
                if min_distance >= self.M[i, j] and self.limit_size_cluster(j, i) <= limit_size_percen and self.volumen(j, i) <= volumen_percen:    
                    min_distance = self.M[i, j]
                    d = self.M[i, j]
                    p = (i, j)
        
        if p:
            for j in range(len(self.M)):
                self.M[p[1], j] = min(self.M[p[0], j], self.M[p[1], j])
        
            self.rows_del.add(p[0])
        return d, p

    def merge_minimal_distance_with_queue_treap(self, limit_size_percen, volumen_percen):
        k = None
        p = None


        for dist_info, (i, j) in self.priorityqueue_treap.items():
           
            if self.row_del_queue[i]:
                print("continue_row: ", i )
                continue
            
            if self.limit_size_cluster(j, i) <= limit_size_percen and self.volumen(j, i) <= volumen_percen:
                k = dist_info
                p = (i, j)
                break

        if k != None:
            self.row_del_queue[p[0]] = True
            k = k[0]
            
        return k, p

    def merge_minimal_distance_with_queue(self, limit_size_percen, volumen_percen):
        k = None
        p = None
        for dist_merge, (i, j) in self.priorityqueue:
           
            if self.row_del_queue[i]:
                continue

            if self.limit_size_cluster(j, i) <= limit_size_percen and self.volumen(j, i) <= volumen_percen:
                k = dist_merge  
                p = (i, j)
                break

        if k != None:
            self.row_del_queue[p[0]] = True
        
        return k, p

    def merge_cluster(self, limit_size_percen, volumen_percen):
        
        if self.total_cluster == 1:
            #print("M:", self.M.shape)
            return False
        
        d, p = self.merge_minimal_distance_with_queue(limit_size_percen, volumen_percen)
        if not p:
            return False

        if len(self.cluster) == 0:
            for e in range(len(self.M)):
                if not set(p).intersection({e}):
                    self.cluster.add(frozenset({e}))
            
            self.cluster.add(frozenset(p))
        else:
            
            temp = set()
            u = set()
            for e in self.cluster:
                if e.intersection(p):
                    u = u.union(e).union(p)
                else:
                    temp.add(e)
            if u:
                temp.add(frozenset(u))
                
            if len(self.cluster) <= len(temp):
                return False

            self.cluster = temp.copy()

        self.cluster_iteration.append(self.cluster.copy())
        self.total_cluster = len(self.cluster)
        self.distance.append(d)
        self.position.append(p)
        self.step += 1
        return True
 
def check_binary(string):
    try:
        int(string, 2)
    except ValueError:
        return False
    return True

def text_to_numpy(all_solutions, params):
    data = []
    solutions = []
    for line in all_solutions:
        if not line:
            continue
        name, it, fitness, *vector_solution = line.split(',')
        data.append([name, int(it), float(fitness)])
        if params.typeproblem == "discrete":
        #if distance_method == "hamming" or distance_method == "euclidean" or distance_method == "manhattan":
            data.append([name, int(it), str(vector_solution[1])])
            solutions.append(bytearray(map(ord, vector_solution[0])))
            solutions.append(bytearray(map(ord, vector_solution[2])))
       
        else:
            solutions.append([float(e) for e in vector_solution])
        
    return np.array(data), np.array(solutions)


def get_hashes_cluster(S, clusters):

    hashes = {}
    for i, c in enumerate(clusters):
        for row in c:
            hashes[row] = sha256(str(i).encode('utf-8')).hexdigest()
    return hashes

class AgglomerativeConfig(object):
    def __init__(self, clustering, number_of_clusters) -> None:
        self.clustering = clustering
        self.number_of_clusters = number_of_clusters


def continuous_agglomerative(params, cfiles):
    
    info, S = text_to_numpy(cfiles, params)
    M = np.zeros((len(S), len(S)), dtype=object)

    print("params.agglomerative_clustering.cluster_size: ", params.agglomerative_clustering.cluster_size)
    print("params.agglomerative_clustering.volumen_size: ", params.agglomerative_clustering.volumen_size)

    C = Clustering(S, M, params.typeproblem, params.agglomerative_clustering.cluster_size, params.agglomerative_clustering.volumen_size, params.agglomerative_clustering.distance_method)
    clusters = sorted([len(c) for c in C.cluster_iteration])
    
    if params.typeproblem == "discrete":
        if len(clusters) > 500:
            clusters = clusters[:500]
        min_clusters = max(min(clusters), 10)
        
    else:
        pos_cluster = math.ceil(len(clusters)/3)
        if pos_cluster <= len(clusters):
            min_clusters = clusters[math.ceil(len(clusters)/3)]
        else:
            min_clusters = clusters[0]

    algorithms = np.unique(info[:, 0])

    results = { algo: AgglomerativeConfig([], []) for algo in algorithms }

    for clusters in C.cluster_iteration:
        if len(clusters) < min_clusters:
            continue
        hashes = get_hashes_cluster(S, clusters)
        for algo in algorithms:
            result = ["Run,Fitness1,Solution1,Fitness2,Solution2"]
            idx = 1
            for i in range(len(S) - 1):
                if info[i][0] != algo:
                    continue
                    
                if idx != info[i+1][1]:
                    idx = info[i+1][1]
                    continue

                it = info[i][1]
                fitnnes1 =  info[i][2]
                sol1 = hashes[i]
                fitnnes2 =  info[i+1][2]
                sol2 = hashes[i+1]

                result.append("{},{},{},{},{}".format(it, fitnnes1, sol1, fitnnes2, sol2))
            results[algo].clustering.append(result)
            results[algo].number_of_clusters.append(len(clusters))
    print("Min clusters:", min_clusters)
    return results, min_clusters
