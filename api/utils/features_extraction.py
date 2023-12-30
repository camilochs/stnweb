from collections import Counter
from utils.gpt import get_template
import numpy as np
import os 
from pprint import pprint

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

    for algorithm, data in info.items():
        text += f"\t - {algorithm} has {data['total_best_fitness']} nodes pointing to the best fitness. \n"
        text += f"\t - {algorithm} has {data['total_merge_local']} percentage of overlapping among all the nodes. \n"
        text += f"\t - {algorithm} among its {data['total_trajectory']} trajectories averages {data['mean_last_fitness']} among all its fitness. \n"
    return text


def information_extraction(all_data, params):
   
    for number_cluster, algorithm_graphs in all_data.items():
        fitness = set()
        for d in algorithm_graphs.values():
            for node in d:
                fitness.add(node.fitness)

        best_fitness = min(fitness)

        info = {}

        for algorithm in algorithm_graphs:
            if algorithm not in info:
                info[algorithm] = {}
            info[algorithm]["total_best_fitness"] = sum([1 for e in algorithm_graphs[algorithm] if e.fitness == best_fitness])
            info[algorithm]["total_trajectory"] = max([e.it for e in algorithm_graphs[algorithm]])
            #print(np.mean(list(Counter([e.cluster_hash for e in algorithm_graphs[algorithm]]).values())))
            info[algorithm]["total_merge_local"] = round(np.mean([e for e in Counter([e.cluster_hash for e in algorithm_graphs[algorithm]]).values()]), 2)
            info[algorithm]["mean_last_fitness"] = round(np.mean([ [node for node in algorithm_graphs[algorithm] if node.it == i][-1].fitness for i in range(1, max([e. it for e in algorithm_graphs[algorithm]])) ]), 3)
            #print(algorithm, number_cluster, info[algorithm])
            #input()
        
        query = get_template()

        new_folder = f"temp/{params.hash_file}"
        if not os.path.exists(f"temp/{params.hash_file}"):
            os.makedirs(new_folder)
        """with open(f'{new_folder}/features-{number_cluster}-context.txt', 'w') as fp:
            prompt_context = context.replace("{type_problem}", 'minimizing' if params.bmin == 1 else 'maximizing')
            prompt_context = prompt_context.replace("{min_cluster}", str(min(all_data.keys())))
            prompt_context = prompt_context.replace("{max_cluster}", str(max(all_data.keys())))
            fp.write(prompt_context)"""

        with open(f'{new_folder}/features-{number_cluster}-query.txt', 'w') as fp:
            prompt_query = query.replace("{info_algorithm}", build_info_by_algorithm(info))
            prompt_query = prompt_query.replace("{type_problem}", 'minimizing' if params.bmin == 1 else 'maximizing')
            prompt_query = prompt_query.replace("{agglomerative_params}", build_info_agglomerative(all_data, params))
            prompt_query = prompt_query.replace("{min_cluster}", str(min(all_data.keys())))
            prompt_query = prompt_query.replace("{max_cluster}", str(max(all_data.keys())))

            fp.write(prompt_query)

    