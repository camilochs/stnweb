from collections import Counter
from utils.gpt import get_template
import numpy as np
import os 
import networkx as nx
import itertools


def build_info_agglomerative(all_solutions, params):
    if params.agglomerative_clustering.number_of_clusters == None:
        params.agglomerative_clustering.number_of_clusters = min(all_solutions.keys())

    return f""" 
    \t- cluster size: {params.agglomerative_clustering.cluster_size}
    \t- volumen size: {params.agglomerative_clustering.volumen_size}
    \t- distance method: {params.agglomerative_clustering.distance_method}
    """ + "\t- cluster_number: {number_of_clusters}"

def build_info_by_algorithm(info):
    text = "\n"

    for algorithm, data in sorted(info.items()):
        text += f"\t\t - {algorithm} has {data['total_best_fitness']} nodes pointing to the best fitness. \n"
        text += f"\t\t - {algorithm} has {data['total_merge_local']} percentage of overlapping among all its nodes. \n"
        text += f"\t\t - {algorithm} among its {data['total_trajectory']} trajectories averages fitness values {data['mean_last_fitness']}. \n"
    return text


def get_connected_graph(algorithm_data, max_trajectories):
    
    graphs = []
    for i in range(0, max_trajectories):
        graphs.append(nx.Graph())

    trajectory = 1
    i = -1
    while i < len(algorithm_data) - 2:
        i += 1
        if algorithm_data[i].it == trajectory and algorithm_data[i + 1].it == trajectory:
            graphs[trajectory - 1].add_edge(algorithm_data[i].cluster_hash, algorithm_data[i + 1].cluster_hash)
        else:
            trajectory += 1

    combinatoric_trajectories = list(itertools.combinations(list(range(0, max_trajectories)), 2))

    connectivity = 0
    for t1, t2 in combinatoric_trajectories:
        if(len(set(graphs[t1].nodes) & set(graphs[t2].nodes)) > 0):
            connectivity += 1

    #print("connectivity: ", connectivity / len(combinatoric_trajectories))
    #input()
    return connectivity

def information_extraction(all_data, params, min_clusters, max_cluster):
   
    for number_cluster, algorithm_graphs in all_data.items():
        fitness = set()
        for d in algorithm_graphs.values():
            for node in d:
                fitness.add(node.fitness)
        
        best_fitness = min(fitness) if params.bmin == 1 else max(fitness)
      

        info = {}

        for algorithm in algorithm_graphs:
            if algorithm not in info:
                info[algorithm] = {}
            
            info[algorithm]["total_best_fitness"] = sum([1 for e in algorithm_graphs[algorithm] if e.fitness == best_fitness])
            info[algorithm]["total_trajectory"] = max([e.it for e in algorithm_graphs[algorithm]])
            mean_values= [ [node for node in algorithm_graphs[algorithm] if node.it == i][-1].fitness for i in range(1, max([e. it for e in algorithm_graphs[algorithm]])) ]
            if mean_values:
                info[algorithm]["mean_last_fitness"] = round(np.mean(mean_values), 3)
            else:
                info[algorithm]["mean_last_fitness"] = algorithm_graphs[algorithm][-1].fitness
 
            info[algorithm]["total_merge_local"] = get_connected_graph(algorithm_graphs[algorithm], info[algorithm]["total_trajectory"] )
        
        template_general = get_template()


        new_folder = f"temp/{params.hash_file}"
        if not os.path.exists(f"temp/{params.hash_file}"):
            os.makedirs(new_folder)
        """with open(f'{new_folder}/features-{number_cluster}-context.txt', 'w') as fp:
            prompt_context = context.replace("{type_problem}", 'minimizing' if params.bmin == 1 else 'maximizing')
            prompt_context = prompt_context.replace("{min_cluster}", str(min(all_data.keys())))
            prompt_context = prompt_context.replace("{max_cluster}", str(max(all_data.keys())))
            fp.write(prompt_context)"""

        with open(f'{new_folder}/features-{number_cluster}-query.txt', 'w') as fp:
            prompt_query = template_general.replace("{{features}}", build_info_by_algorithm(info))
            prompt_query = prompt_query.replace("{{type_problem}}", 'minimizing' if params.bmin == 1 else 'maximizing')
            #prompt_query = prompt_query.replace("{agglomerative_params}", build_info_agglomerative(all_data, params))
            #prompt_query = prompt_query.replace("{min_cluster}", str(min(all_data.keys())))
            #prompt_query = prompt_query.replace("{max_cluster}", str(max(all_data.keys())))

            fp.write(prompt_query)

    