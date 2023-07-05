from typing import Dict
from scipy.stats import entropy
import math
import operator
from hashlib import sha256
from pprint import pprint

class Stats:
        pos = 0
        one_freq = 0.0
        entropy = 0.0

def get_total_area(totalNodes, stats):
        total_area = 0.0

        for i in range(1, totalNodes):
                total_area += stats[i].entropy + ((stats[i-1].entropy - stats[i].entropy)/2.0)
        return total_area

def standard(datainput, c, all_solutions):
        print("Start partitions")
        
        partitions_one_algorithm = set()
        partitions = set()
        partitions_idx_noset : Dict[int, tuple] = {}
        allFitness : Dict[str, int] = {}
        for line in datainput:
            if not line:
                    continue
            idx, fitness1, bin1, fitness2, bin2 = line.split(',')
            if idx == "Run":
                    continue
            idx = int(idx)
            bin1 = bin1.rstrip()
            bin2 = bin2.rstrip()
            if idx not in partitions_idx_noset:
                    partitions_idx_noset[idx] = []

            partitions_idx_noset[idx].append((int(fitness1), bin1))
            partitions_idx_noset[idx].append((int(fitness2), bin2))
            
            
            partitions_one_algorithm.add(bin1)
            partitions_one_algorithm.add(bin2)

        for line in all_solutions:
            if not line:
                    continue
            _, fitness1, bin1, fitness2, bin2 = line.split(',')
            partitions.add(bin1)
            partitions.add(bin2)
            allFitness[bin1] = int(fitness1)
            allFitness[bin2] = int(fitness2)

        solutions = list(partitions_one_algorithm)

        all_solutions = partitions.copy()
        totalNodes = len(solutions[0])
        stats = []

        for i in range(totalNodes):
                stats.append(Stats())
                stats[i].pos = i

        for i in range(totalNodes):
                stats[i].one_freq = float([b[i] for b in all_solutions].count('1'))


        for i in range(totalNodes):
                stats[i].one_freq /= float(len(all_solutions))
                if stats[i].one_freq > 0.0:
                        stats[i].entropy += (stats[i].one_freq * math.log2(stats[i].one_freq))
                if ((1.0 - stats[i].one_freq) > 0.0):
                        stats[i].entropy += ((1.0 - stats[i].one_freq) * math.log2(1.0 - stats[i].one_freq))
                
                stats[i].entropy *= -1.0
        stats = sorted(stats, key=operator.attrgetter('entropy'), reverse=True)
        results = ["Run,Fitness1,Solution1,Fitness2,Solution2"]
        totalArea = get_total_area(totalNodes, stats)
        
        totalVars = int(math.floor((totalNodes*c)/100.0))
        if totalVars == 0:
                totalVars = 1
        print("total vars:", totalVars)
        print("C: ------------> {} ".format(c))

        repr : Dict[str, str]= {}
        fitness : Dict[str, int] = {}
        print("n_vars: ", totalVars)
        for sol in all_solutions:
                
                cSols = [] 
                
                for _ in range(totalVars):
                        cSols.append("0")
                for j in range(totalVars):
                        cSols[j] = sol[stats[j].pos]
        
                cSols = ''.join(cSols)
                repr[sol] = cSols

                if cSols not in fitness:
                        fitness[cSols] = allFitness[sol]
                else:
                        if allFitness[sol] < fitness[cSols]:
                                fitness[cSols] = allFitness[sol]
        
        for k, data in partitions_idx_noset.items():
                bin = [e[1] for e in data]
                for i in range(0, len(bin)-1, 2):
                        if c > 0 and (repr[bin[i]] != repr[bin[i+1]]):
                          sol1 = sha256(repr[bin[i]].encode('utf-8')).hexdigest()
                          sol2 = sha256(repr[bin[i+1]].encode('utf-8')).hexdigest()
                          results.append("{},{},{},{},{}".format(k, fitness[repr[bin[i]]], sol1, fitness[repr[bin[i+1]]], sol2))  
                        elif c == 0:
                          sol1 = sha256(bin[i].encode('utf-8')).hexdigest()
                          sol2 = sha256(bin[i+1].encode('utf-8')).hexdigest()
                          results.append("{},{},{},{},{}".format(k, allFitness[bin[i]], sol1, allFitness[bin[i+1]], sol2))     
        
        return results




def partition_old(datainput, c, all_solutions):
        print("Start partitions")
        
        partitions_one_algorithm = set()
        partitions = set()
        partitions_idx_noset : Dict[int, tuple] = {}
        allFitness : Dict[str, int] = {}
        for line in datainput:
            if not line:
                    continue
            idx, fitness1, bin1, fitness2, bin2 = line.split(',')
            if idx == "Run":
                    #print(','.join(data))
                    continue
            idx = int(idx)
            bin1 = bin1.rstrip()
            bin2 = bin2.rstrip()
            if idx not in partitions_idx_noset:
                    partitions_idx_noset[idx] = []

            partitions_idx_noset[idx].append((int(fitness1), bin1))
            partitions_idx_noset[idx].append((int(fitness2), bin2))
            
            
            partitions_one_algorithm.add(bin1)
            partitions_one_algorithm.add(bin2)

        for line in all_solutions:
            if not line:
                    continue
            _, fitness1, bin1, fitness2, bin2 = line.split(',')
            partitions.add(bin1)
            partitions.add(bin2)
            allFitness[bin1] = int(fitness1)
            allFitness[bin2] = int(fitness2)

        solutions = list(partitions_one_algorithm)

        all_solutions = partitions.copy()
        totalNodes = len(solutions[0])
        stats = []

        for i in range(totalNodes):
                stats.append(Stats())
                stats[i].pos = i

        for i in range(totalNodes):
                stats[i].one_freq = float([b[i] for b in all_solutions].count('1'))


        for i in range(totalNodes):
                stats[i].one_freq /= float(len(all_solutions))
                #print("one_freq:", i, stats[i].one_freq, float(len(all_solutions)))
                if stats[i].one_freq > 0.0:
                        stats[i].entropy += (stats[i].one_freq * math.log2(stats[i].one_freq))
                if ((1.0 - stats[i].one_freq) > 0.0):
                        stats[i].entropy += ((1.0 - stats[i].one_freq) * math.log2(1.0 - stats[i].one_freq))
                
                stats[i].entropy *= -1.0

        stats = sorted(stats, key=operator.attrgetter('entropy'), reverse=True)

        results = ["Run,Fitness1,Solution1,Fitness2,Solution2"]
        totalArea = get_total_area(totalNodes, stats)

        
        coarsening = [1.0 - (c/100)]
        totalVars = int(c)
        print("C: ------------> {} ".format(c))
        for i in range(len(coarsening)):
      
                repr : Dict[str, str]= {}
                fitness : Dict[str, int] = {}
                for sol in all_solutions:
                        
                        cSols = [] 
                        
                        for _ in range(totalVars):
                                cSols.append("0")
                        for j in range(totalVars):
                                cSols[j] = sol[stats[j].pos]
                
                        cSols = ''.join(cSols)
                        repr[sol] = cSols

                        if cSols not in fitness:
                                fitness[cSols] = allFitness[sol]
                        else:
                                if allFitness[sol] < fitness[cSols]:
                                        fitness[cSols] = allFitness[sol]
                
                for k, data in partitions_idx_noset.items():
                        bin = [e[1] for e in data]
                        for i in range(0, len(bin)-1, 2):
                                if c > 0.0 and (repr[bin[i]] != repr[bin[i+1]]):
                                        results.append("{},{},{},{},{}".format(k, fitness[repr[bin[i]]], repr[bin[i]], fitness[repr[bin[i+1]]], repr[bin[i+1]]))  
                                elif c == 0.0:
                                        results.append("{},{},{},{},{}".format(k, allFitness[bin[i]], bin[i], allFitness[bin[i+1]], bin[i+1]))     
                
  
        return results
