from collections import Counter
import numpy as np
import os 
from pprint import pprint

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
        
        template = """

        These are the rules of the system:

        - Rule 1: The more nodes pointing to the best fitness and with average fitness better (this is a {type_problem} problem), the higher the algorithm's quality because it can find the best result.
        - Rule 2: The algorithm that has more overlap (merge) is likely to be more robust. If and only if there is best fitness.

        Now, considering these rules, take a look at this data: {info_algorithm}

        Finally, give me a general interpretation that allows comparing both algorithms and determining which one is better. 
        
        Important:
         - You response must has a limit to 300 tokens with details.   
         - You response must be in HTML format.
         - It is forbidden for the response to be in markdown format.
         - In the answer add bold the name of each algorithm.
         - This is a {type_problem} optimization problem. 

        """

        new_folder = f"temp/{params.hash_file}"
        if not os.path.exists(f"temp/{params.hash_file}"):
            os.makedirs(new_folder)
        with open(f'{new_folder}/features-{number_cluster}.txt', 'w') as fp:
            prompt = template.replace("{type_problem}", 'minimizing' if params.bmin == 1 else 'maximizing')
            prompt = prompt.replace("{info_algorithm}", build_info_by_algorithm(info))
            fp.write(prompt)

    