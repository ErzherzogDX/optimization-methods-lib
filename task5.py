import math

import random


def get_distance(points, path):
    return sum(math.dist(points[path[i - 1]], points[path[i]]) for i in range(len(path)))

def generate_random_paths(total_destinations, count_of_perms):
    random_paths = []
    for _ in range(count_of_perms):
        random_path = list(range(1, total_destinations))
        random.shuffle(random_path)
        random_path = [0] + random_path
        random_paths.append(random_path)
    return random_paths


def choose_survivors(old_generation, size_of_survivors):
    return random.sample(old_generation, size_of_survivors)


def born_descendants(parent_a, parent_b):
    offspring = []
    start = random.randint(0, len(parent_a) - 1)
    finish = random.randint(start, len(parent_a))
    sub_path_from_a = parent_a[start:finish]
    remaining_path_from_b = list([item for item in parent_b if item not in sub_path_from_a])
    for i in range(0, len(parent_a)):
        if start <= i < finish:
            offspring.append(sub_path_from_a.pop(0))
        else:
            offspring.append(remaining_path_from_b.pop(0))
    return offspring


def get_crossovers(survivors, prob_crossover):
    descendants = []
    to_cross = False
    if (random.randint(0, 100) / 100) < prob_crossover: to_cross = True
    midway = len(survivors) // 2
    for i in range(midway):
        parent_a, parent_b = survivors[i], survivors[i + midway]
        if to_cross:
            descendants.append(born_descendants(parent_a, parent_b))
            descendants.append(born_descendants(parent_b, parent_a))
        else:
            descendants.append(parent_a)
            descendants.append(parent_b)
    return descendants


def get_mutations(generation, prob_mutation):
    mutated = []
    for path in generation:
        if (random.randint(0, 100) / 100) < prob_mutation:
            ix1, ix2 = random.randint(1, len(path) - 1), random.randint(1, len(path) - 1)
            path[ix1], path[ix2] = path[ix2], path[ix1]
        mutated.append(path)
    return mutated


def generate_new_population(points, old_generation, num_survivors, prob_mutation, prob_cross):
    pairs = [[lst, get_distance(points, lst)] for lst in old_generation]
    sorted_pairs = sorted(pairs, key=lambda x: x[1])

    survivors = choose_survivors(old_generation, num_survivors)
    crossovers = get_crossovers(survivors, prob_cross)
    new_population = get_mutations(crossovers, prob_mutation)

    sorted_pairs = sorted_pairs[:-num_survivors]
    list_of_lists = [pair[0] for pair in sorted_pairs]
    list_of_lists += new_population
    return list_of_lists


if __name__ == '__main__':
    print("Enter N (population), M (survivors), mu (prob. of mutation) and nu (prob. of crossovers)")
    size = 20
    total_iterations = 5000
    ev_populations = []
    coords = [(1234, 567), (456, 789), (234, 876), (987, 543), (345, 123),
              (678, 234), (890, 678), (234, 456), (789, 123), (567, 890),
              (432, 210), (654, 321), (876, 543), (210, 654), (543, 876),
              (321, 432), (987, 210), (210, 987), (654, 789), (789, 210)]

    N = int(input())
    M = int(input())
    mu = float(input())
    nu = float(input())

    # N = 100
    # M = 20
    # mu = 0.1
    # nu = 0.9

    ev_populations.append(generate_random_paths(size, N))
    for _ in range(total_iterations):
        ev_populations.append(generate_new_population(coords, ev_populations[-1], M, mu, nu))

    mins = []
    for p in range(total_iterations):
        population = ev_populations[p]
        minimum = math.inf
        for path in population:
            minimum = min(minimum, get_distance(coords, path))
        mins.append(minimum)
        print("Iteration ", p, ": ", minimum)

    mins = sorted(mins)
    print("Finded min.dist. by algorithm is", N, "points:", mins[:N])
