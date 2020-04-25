import datetime
import random
import unittest
import copy

def _generate_parent(length, geneSet, get_fitness):
    chromosome_list = []
    for i in range(0, 10):
        genes = []
        while len(genes) < length:
            sampleSize = min(length - len(genes), len(geneSet))
            genes.extend(random.sample(geneSet, sampleSize))
        fitness = get_fitness(genes)
        chromosome_list.append(Chromosome(genes, fitness))
    return chromosome_list

def _mutate(parent, geneSet, get_fitness):
    childGenes = parent.Genes[:]
    print(childGenes)    
    index = random.randrange(0, len(parent.Genes))
    newGene, alternate = random.sample(geneSet, 2)
    childGenes[index] = alternate if newGene == childGenes[index] else newGene
    print(index, newGene, alternate)
    print(childGenes)
    fitness = get_fitness(childGenes)
    return Chromosome(childGenes, fitness)

def _generate_child(parent_list, geneSet, get_fitness):
    child_list = []
    fitness_percent_list = []
    fitness_accum_list = []
    fitness_sum = 0
    for parent in parent_list:
        fitness_sum += parent.Fitness

    for parent in parent_list:
        fitness_percent_list.append(parent.Fitness / fitness_sum)

    fitness_sum = 0
    for fitness_percent in fitness_percent_list:
        fitness_sum += fitness_percent
        fitness_accum_list.append(fitness_sum)

    # Selection
    for i in range(0, 10):
        rand = random.random()
        before = 0
        for j in range(0, len(fitness_accum_list)):
            if rand > before and rand <= fitness_accum_list[j]:
                child_list.append(copy.deepcopy(parent_list[j]))
                break
            before = fitness_accum_list[j]

    # Crossover
    crossover_rate = 0.20
    selected = None
    for i in range(0, len(child_list)):
        rand = random.random()
        if rand < crossover_rate:
            if selected is None:
                selected = i
            else:
                child_list[selected].Genes[2:], child_list[i].Genes[2:] = \
                    child_list[i].Genes[2:], child_list[selected].Genes[2:]
                selected = None

        # update
        child_list[i].Fitness = get_fitness(child_list[i].Genes)

    # mutate
    mutate_rate = 0.2
    for i in range(0, len(child_list)):
        rand = random.random()
        if rand < mutate_rate:
            child = _mutate(child_list[i], geneSet, get_fitness)
            del child_list[i]
            child_list.append(child)
    return child_list


def get_best(get_fitness, targetLen, optimalFitness, geneSet, display):
    random.seed()

    # 1. Generate Parent
    bestParentList = _generate_parent(targetLen, geneSet, get_fitness)
    display(bestParentList)

    gen_count = 0
    maximum_average = 0
    while True:
        gen_count += 1
        #print("generation : {}".format(gen_count))
        child_list = _generate_child(bestParentList, geneSet, get_fitness)

        fitness_sum = 0
        for child in child_list:
            fitness_sum += child.Fitness

        average = fitness_sum / 10
        if average > maximum_average:
            print("new maximum fitness : {}".format(average))
            bestParentList = child_list
            maximum_average = average

        if average >= optimalFitness:
            return child_list

class Chromosome:
    def __init__(self, genes, fitness):
        self.Genes = genes
        self.Fitness = fitness

def get_fitness(guess, target):
    fitness = 0
    for expected, actual in zip(target, guess):
        if expected == actual:
            fitness += 5
        elif actual in target:
            fitness += 1

    return fitness

def display(candidate, target, startTime):
    timeDiff = datetime.datetime.now() - startTime
    strike = 0
    ball = 0
    for expected, actual in zip(target, candidate.Genes):
        if expected == actual:
            strike += 1
        elif actual in target:
            ball += 1

    if strike == 0 and ball == 0:
        result = "out"
    else:
        result = "{}/{}".format(strike, ball)

    print("{}\t{}\t{}\t{}".format(
        ''.join(candidate.Genes),
        result,
        candidate.Fitness,
        timeDiff))

def display_list(candidate_list, target, startTime):
    fitness_sum = 0
    for candidate in candidate_list:
        display(candidate, target, startTime)
        fitness_sum += candidate.Fitness
    print("average fitness : {}".format(fitness_sum / len(candidate_list)))

def pick_baseball_num(length, is_duplicate_allowed):
    if is_duplicate_allowed is True or length > 10:
        return ''.join(random.choice("0123456789")
                       for _ in range(length))

    baseball_list = []
    num = random.randrange(0, 10)

    for i in range(length):
        while str(num) in baseball_list:
            num = random.randrange(0, 10)
        baseball_list.append(str(num))

    return ''.join(baseball_list)


class BaseballGames(unittest.TestCase):
    geneset = "0123456789"

    def test_Baseball(self):
        length = 6

        target = pick_baseball_num(length, False)
        print("target : {}".format(target))
        self.guess_baseball(target)

    def guess_baseball(self, target):
        startTime = datetime.datetime.now()

        def fnGetFitness(genes):
            return get_fitness(genes, target)

        def fnDisplay(candidate_list):
            display_list(candidate_list, target, startTime)

        optimalFitness = len(target) * 5
        child_list = get_best(fnGetFitness, len(target), optimalFitness,
                                self.geneset, fnDisplay)

        fnDisplay(child_list)


if __name__ == '__main__':
    unittest.main()