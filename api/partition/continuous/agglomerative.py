
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
import random
from itertools import product
from multiprocessing import Pool, Lock
from pprint import pprint
from ordered_set import OrderedSet

class Node(object):

    def __init__(self, it, idx_line, fitness, solution):
        self.it = it
        self.idx_line = idx_line
        self.fitness = fitness
        self.solution = solution

    def __repr__(self):
        return f"Node({self.it}, {self.idx_line}, {self.fitness}, {self.solution})"
    def __eq__(self, other):
        if isinstance(other, Node):
            return ((self.idx_line == other.idx_line) and (self.it == other.it) and (self.fitness == other.fitness) and (self.solution == other.solution) )
        else:
            return False
            
    def __hash__(self):
        return hash((self.idx_line, self.it, self.fitness, self.solution))
        


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

        for e in range(len(self.M)):
                self.cluster.add(frozenset({e}))
        self.cluster_iteration.append(self.cluster.copy())
        self.cluster = set()
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
                    """if self.M[i,j] != 0:
                        print(i, j,self.M[i,j], len(self.S[i]), _i, _j) 
                        input()"""
                heapq.heappush(self.priorityqueue, (self.M[i, j],  (i, j)))
                sum_dist += self.M[i,j]
            print(f"complete (processing distances): { i * 100 / len(self.S):.2f} %", end='\r')

        if distance_method == 'hamming':
            self.hamming_cache.clear()
        self.priorityqueue = sorted(self.priorityqueue)
        
        
        #pprint(self.M)
        print(*self.M, sep="\n")
        pprint(self.M.shape)
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
            return False

        d, p = self.merge_minimal_distance_with_queue(limit_size_percen, volumen_percen)
        if not p:
            return False

        if len(self.cluster_iteration) == 1:
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
    
    data_nodes = {}
    index_line = 0
    for line in all_solutions:
        name, it, fitness, *data_next_node = line.split(',')
        if name not in data_nodes:
            if len(data_nodes) > 0: # keeps the correlative of the line index
                index_line += 1
            data_nodes[name] = OrderedSet()
        else:
            if data_nodes[name][-1].it != int(it): # keeps the correlative of the line index
                index_line += 1

        if params.typeproblem == "discrete":
            data_nodes[name].add(Node(int(it), index_line, float(fitness), data_next_node[0]))
            data_nodes[name].add(Node(int(it), index_line+1, float(data_next_node[1]), data_next_node[2]))
       
        else:
            data_nodes[name].add(Node(int(it), index_line, float(fitness), [float(elem) for elem in data_next_node[0]]))
            data_nodes[name].add(Node(int(it), index_line+1, float(data_next_node[1]), [float(elem) for elem in data_next_node[2]]))
        index_line += 1

    solutions = [bytearray(map(ord, node.solution)) for algorithm in data_nodes for node in data_nodes[algorithm]]
    print("len: ", len(all_solutions)*2)
    return data_nodes, np.array(solutions), np.array(solutions)


def get_hashes_cluster(clusters, data_nodes):
    import itertools

    hashes = {}
    nodes = list(itertools.chain.from_iterable([e for e in data_nodes.values()]))
 
    for i, c in enumerate(clusters):
        choice_cluster = max(c)
        for row in c:
            #print("-----", len(clusters), len(c), i, row, len(nodes), choice_cluster)
            #input()
            hashes[row] = sha256(str(nodes[choice_cluster].solution).encode('utf-8')).hexdigest()
    return hashes

class AgglomerativeConfig(object):
    def __init__(self, clustering, number_of_clusters) -> None:
        self.clustering = clustering
        self.number_of_clusters = number_of_clusters


def continuous_agglomerative(params, cfiles):
    
    data_nodes, S, S_post_processing = text_to_numpy(cfiles, params)
    M = np.zeros((len(S_post_processing), len(S_post_processing)), dtype=object)
    print("S: ", len(S))
    print("M: ", len(M))
    print("S_post: ", len(S_post_processing))
    print("params.agglomerative_clustering.cluster_size: ", params.agglomerative_clustering.cluster_size)
    print("params.agglomerative_clustering.volumen_size: ", params.agglomerative_clustering.volumen_size)

    C = Clustering(S_post_processing, M, params.typeproblem, params.agglomerative_clustering.cluster_size, params.agglomerative_clustering.volumen_size, params.agglomerative_clustering.distance_method)
    C.cluster_iteration = [c for c in C.cluster_iteration]
    clusters = sorted([len(c) for c in C.cluster_iteration])
    
    print("min clusters:", min(clusters))
    print("max clusters:", max(clusters))
    if params.typeproblem == "discrete":
        min_clusters = max(min(clusters), 10)
        
    else:
        pos_cluster = math.ceil(len(clusters)/3)
        if pos_cluster <= len(clusters):
            min_clusters = clusters[math.ceil(len(clusters)/3)]
        else:
            min_clusters = clusters[0]


    results = { algo: AgglomerativeConfig([], []) for algo in data_nodes.keys() }
    info_analytics = {}
    aggregation = {}
    for cluster in C.cluster_iteration:
        if len(cluster) < min_clusters:
            continue
        
        hashes = get_hashes_cluster(cluster, data_nodes)
        #for h in sorted(hashes):
        #    print(h, hashes[h])
        #input()
        if len(cluster) not in aggregation:
            aggregation[len(cluster)] = []

        for algorithm, nodes_data in data_nodes.items():
            result = ["Run,Fitness1,Solution1,Fitness2,Solution2"]
            i = 0
            

            while i < (len(nodes_data) - 1):
                if nodes_data[i].it != nodes_data[i+1].it:
                    aggregation[len(cluster)].append(hashes[nodes_data[i].idx_line])
                    i += 1
                    continue
                info_node = "{},{},{},{},{}".format(nodes_data[i].it,
                                                    nodes_data[i].fitness,
                                                    hashes[nodes_data[i].idx_line],
                                                    nodes_data[i+1].fitness,
                                                    hashes[nodes_data[i+1].idx_line])
                aggregation[len(cluster)].append(hashes[nodes_data[i].idx_line])
                #print(info_node)
                result.append(info_node)
                
                i += 1
                #print(info_node)
                #input()
            aggregation[len(cluster)].append(hashes[nodes_data[-1].idx_line])
            #input()
            results[algorithm].clustering.append(result)
            results[algorithm].number_of_clusters.append(len(cluster))
            #print(f"clustering {algo}: ", results[algo].clustering)
            #print(f"number of clusters {algo}: ", results[algo].number_of_clusters)
            #input()
    for cluster_size, v in aggregation.items():
        if cluster_size != 237 and cluster_size != 175 and cluster_size != 120:
            continue
        agg = Counter(v)
        agg = Counter(agg.values())
        suma = 0
        #input()
        for k in sorted(agg):
            for _ in range(agg[k]):
                print(k, end=",")
            suma += (k*agg[k])
        print()
        print(cluster_size, suma)
        #input()
        print("----")
    return results, min_clusters

