import numpy as np

def text_to_numpy(all_solutions):
    data = []
    solutions = []
    for line in all_solutions:
        if not line:
            continue
        name, it, fitness, *vector_solution = line.split(',')
        data.append([name, int(it), float(fitness)])
        solutions.append([float(e) for e in vector_solution])
    return np.array(data), solutions

def hash_solution(solutions, min_bound, max_bound, pf):

    p = np.arange(min_bound, max_bound, pf).tolist()
    for i in range(len(solutions)):
        for j in range(len(solutions[i])):
            for k in range(len(p) - 1):
                if(p[k] <= solutions[i][j] and solutions[i][j] <= p[k+1]):
                    solutions[i][j] = int(k)
                    break
        solutions[i] = hash(str(solutions[i]))
    
    return solutions

def continuous_standard(params, cfiles):

    info, S = text_to_numpy(cfiles)
    S = hash_solution(S, params.standard_configuration.min_bound, params.standard_configuration.max_bound, params.standard_configuration.partition_factor)

    algorithms = np.unique(info[:, 0])
    results = { algo: [] for algo in algorithms }

    idx = 1
    for algo in algorithms:
        result = ["Run,Fitness1,Solution1,Fitness2,Solution2"]
        for i in range(len(S) - 1):
            if info[i][0] != algo:
                continue
            if idx != info[i+1][1]:
                idx = info[i+1][1]
                continue
            it = info[i][1]
            fitnnes1 =  info[i][2]
            sol1 = S[i]
            fitnnes2 =  info[i+1][2]
            sol2 = S[i+1]

            result.append("{},{},{},{},{}".format(it, fitnnes1, sol1, fitnnes2, sol2))
        results[algo].append(result)
    
    return results
