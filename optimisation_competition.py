###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from assignment1_test_controller import player_controller
from optimization_specialist_assignment1 import sigma_scaling, swap_mutation

# imports other libs
import time
import numpy as np
import random
from math import fabs,sqrt
import glob, os
import gzip, pickle, yaml
import argparse
import wandb
import statistics
import itertools
from nnlayers import tanh_activation, sigmoid_activation, softmax_activation
from functools import total_ordering, reduce
from operator import add
import copy

VERBOSE = False

def verbose_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

np.random.seed(42)
random.seed(42)

@total_ordering
class Individual:

    def __init__(self, 
                 id: int | None = None,
                 fitness: float = 0, 
                 player_energy: float = 0, 
                 enemy_energy: float = 0, 
                 timesteps: int = 0, 
                 gain: float = 0, 
                 dead_enemies: int = 0):
        
        self.wnb: tuple[tuple[np.ndarray, np.ndarray], ...]
        self.fitness = fitness
        self.fitnessalt = fitness
        self.front = 0
        self.crowding_dist = 0.
        self.player_energy = player_energy
        self.enemy_energy = enemy_energy
        self.timesteps = timesteps
        self.gain = gain
        self.dead_enemies = dead_enemies
        self.dominates = set()
        self.dominated_by = 0
        self.id = id if id is not None else random.randint(0, 1000000)
        self.ref_point = 0
        self.ref_dist = 0.

    def intialiseWeights(self, n_hidden_neurons: list[int]) -> None:
        """
        Initializes the weights and biases for a neural network with the given hidden layer sizes.
        Args:
            n_hidden_neurons (list[int]): A list containing the number of neurons in each hidden layer.
        """

        wnblist : list[tuple[np.ndarray, np.ndarray]] = []
        self.n_hidden_neurons = n_hidden_neurons
        in_size = 20 # Amount of input neurons (sensors of the game)
        for layer_size in n_hidden_neurons:
            weights = np.random.uniform(-1,1,(in_size, layer_size))
            bias = np.random.uniform(-1,1,(1, layer_size))
            in_size = layer_size
            wnblist.append((weights, bias))
        self.wnb = tuple(wnblist)

    def setWeights(self, wnb: list[tuple[np.ndarray, np.ndarray]], force = False) -> None:
        """ Set the weights and biases of the individual
        Args:
            wnb (list[tuple[np.ndarray, np.ndarray]]): List of tuples containing the weights and biases for each layer
        """
        if not force:
            assert len(wnb) == len(self.wnb), f"Length of wnb {len(wnb)} does not match the length of the individual's weights and biases {len(self.wnb)}"
            assert all([w[0].shape == wnb[i][0].shape for i, w in enumerate(self.wnb)]), "Shapes of weights do not match"
            assert all([b[1].shape == wnb[i][1].shape for i, b in enumerate(self.wnb)]), "Shapes of biases do not match"
        self.wnb = tuple(wnb)

    def evaluate(self, env: Environment) -> None:
        """Evaluate the individual in the given environment

        Args:
            env (Environment): Environment object for the evoman framework
        """        
        f, p, e, t, de, g, fa = env.play(pcont=self.wnb)
        if self.fitness != 0 and f < self.fitness:
            print(f"Fitness of individual {self.id} decreased from {self.fitness} to {f}")
        self.fitness = f
        self.fitnessarray = fa
        self.player_energy = p
        self.enemy_energy = e
        self.timesteps = t
        self.gain = g
        self.dead_enemies = de
        # self.fitnessalt = 0.6*(100 - e) + 0.4*p - np.log(t)

    def check_domination(self, other: object):
        """Check if this individual dominates another individual

        Args:
            other (Individual): Another individual

        Returns:
            bool: True if this individual dominates the other, False otherwise
        """
        if isinstance(other, Individual):
            gtList = [fs > so for fs, so in zip(self.fitnessarray, other.fitnessarray)]  # Greater than list
            geList = [fs >= so for fs, so in zip(self.fitnessarray, other.fitnessarray)] # Greater or equal list

            if all(geList) and any(gtList): # If all values are greater or equal and at least one is greater, this individual dominates the other
                self.dominates.add(other)
                other.dominated_by += 1
                return True
            return False
        else:
            return NotImplemented

    # Bunch of magic methods to make the Individual class work with the genetic algorithm (if you want to find out more, google function overloading)
    def __str__(self) -> str:
        return f"Ind {self.id}: {self.fitness:.2f}"

    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.id == other.id
    
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
    
    def __lt__(self, other: object) -> bool:
        if isinstance(other, Individual):
            return self.fitness < other.fitness
        elif isinstance(other, (int, float)):
            return self.fitness < other
        return NotImplemented
    
    def __le__(self, other: object) -> bool:
        if isinstance(other, Individual):
            return self.__lt__(other) or self.fitness == other.fitness
        elif isinstance(other, (int, float)):
            return self.fitness <= other
        return NotImplemented
    
    def __gt__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.fitness > other
        elif isinstance(other, Individual):
            return self.fitness > other.fitness
        return NotImplemented
    
    def __ge__(self, other: object) -> bool:
        if isinstance(other, Individual):
            return self.__gt__(other) or self.fitness == other.fitness
        elif isinstance(other, (int, float)):
            return self.fitness >= other
        return NotImplemented
    
    def __hash__(self) -> int:
        return self.id
    
    def __add__(self, other: object) -> float:
        if isinstance(other, Individual):
            return self.fitness + other.fitness
        elif isinstance(other, (int, float)):
            return self.fitness + other
        else:
            return NotImplemented
        
    def __radd__(self, other: object) -> float:
        if isinstance(other, Individual):
            return other.fitness + self.fitness
        elif isinstance(other, (int, float)):
            return other + self.fitness
        else:
            return NotImplemented
        
    def __sub__(self, other: object) -> float:
        if isinstance(other, Individual):
            return self.fitness - other.fitness
        elif isinstance(other, (int, float)):
            return self.fitness - other
        else:
            return NotImplemented
        
    def __rsub__(self, other: object) -> float:
        if isinstance(other, Individual):
            return other.fitness - self.fitness
        elif isinstance(other, (int, float)):
            return other - self.fitness
        else:
            return NotImplemented
        
    def __mul__(self, other: object) -> float:
        if isinstance(other, Individual):
            return self.fitness * other.fitness
        elif isinstance(other, (int, float)):
            return self.fitness * other
        else:
            return NotImplemented
        
    def __rmul__(self, other: object) -> float:
        if isinstance(other, Individual):
            return other.fitness * self.fitness
        elif isinstance(other, (int, float)):
            return other * self.fitness
        else:
            return NotImplemented
        
    def __truediv__(self, other: object) -> float:
        if isinstance(other, Individual):
            return self.fitness / other.fitness
        elif isinstance(other, (int, float)):
            return self.fitness / other
        else:
            return NotImplemented
        
    def __rtruediv__(self, other: object) -> float:
        if isinstance(other, Individual):
            return other.fitness / self.fitness
        elif isinstance(other, (int, float)):
            return other / self.fitness
        else:
            return NotImplemented
        
    def __floordiv__(self, other: object) -> float:
        if isinstance(other, Individual):
            return self.fitness // other.fitness
        elif isinstance(other, (int, float)):
            return self.fitness // other
        else:
            return NotImplemented
        
    def __rfloordiv__(self, other: object) -> float:
        if isinstance(other, Individual):
            return other.fitness // self.fitness
        elif isinstance(other, (int, float)):
            return other // self.fitness
        else:
            return NotImplemented
        
    def __mod__(self, other: object) -> float:
        if isinstance(other, Individual):
            return self.fitness % other.fitness
        elif isinstance(other, (int, float)):
            return self.fitness % other
        else:
            return NotImplemented
        
    def __rmod__(self, other: object) -> float:
        if isinstance(other, Individual):
            return other.fitness % self.fitness
        elif isinstance(other, (int, float)):
            return other % self.fitness
        else:
            return NotImplemented
        
    def __pow__(self, other: object) -> float:
        if isinstance(other, Individual):
            return self.fitness ** other.fitness
        elif isinstance(other, (int, float)):
            return self.fitness ** other
        else:
            return NotImplemented
        
    def __rpow__(self, other: object) -> float:
        if isinstance(other, Individual):
            return other.fitness ** self.fitness
        elif isinstance(other, (int, float)):
            return other ** self.fitness
        else:
            return NotImplemented
        
    def __abs__(self) -> float:
        return abs(self.fitness)
    
    def __neg__(self) -> float:
        return -self.fitness
    
    def __pos__(self) -> float:
        return +self.fitness
    
    def __round__(self, n: int = 0) -> float:
        return round(self.fitness, n)
    
    def __float__(self) -> float:
        return float(self.fitness)
    
    def __int__(self) -> int:
        return int(self.fitness)

# tournament
def tournament(pop: list[Individual], size: int = 2) -> Individual:
    """Tournament function to select the fittest of two individuals in the population

    Args:
        pop (list[Individual): Population of individuals
        size (int, optional): Size of the tournament. Defaults to 2.

    Returns:
        list(tuple(weigths, bias)): One individual in the population
    """
    # Get two random values that correspond to two individuals in the population
    contestants = [pop[random.randint(0, len(pop)-1)] for _ in range(size)]

    return min(contestants, key=lambda x: (x.front, -x.crowding_dist)) # Return the individual with the highest front

##### NSGA-II functions #####

def non_dominant_sorting(pop):
    # Initialize dominance sets
    fronts = [[]]
    # Check for duplicates in pop
    ids = [ind.id for ind in pop]
    if len(ids) != len(set(ids)):
        for i, ind in enumerate(pop):
            ind.id = i
    for ind1 in pop:
        ind1.dominates = set()
        ind1.dominated_by = 0
        for ind2 in pop:
            ind1.check_domination(ind2)
        
    for ind1 in pop:
        if ind1.dominated_by == 0:
            ind1.front = 0
            fronts[0].append(ind1)

    # Perform non-dominant sorting
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for ind1 in fronts[current_front]:
            for ind2 in list(ind1.dominates):
                try:
                    ind2.dominated_by -= 1
                except KeyError:
                    print(f"Individual {ind2.id} does not dominate {ind1.id}. \nDominates: {ind2.dominates}, \nDominated by: {ind2.dominated_by}")
                    pass
                if ind2.dominated_by == 0:
                    ind2.front = current_front + 1
                    next_front.append(ind2)
        current_front += 1
        fronts.append(next_front)

    return fronts[:-1]  # Remove the last empty front

def crowding_distance_sorting(front: list[Individual]) -> list[Individual]:
    for ind in front:
        ind.crowding_dist = 0

    # print(front[0].fitnessarray)
    for m in range(len(front[0].fitnessarray)):
        front.sort(key=lambda x: x.fitnessarray[m])
        front[0].crowding_dist = float('inf')
        front[len(front)-1].crowding_dist = float('inf')
        
        for i in range(1, len(front)-1):
            front[i].crowding_dist += front[i+1].fitness - front[i-1].fitness

    return front

##### NSGA-III functions #####

def make_reference_points(pop: list[Individual], grid_res: int) -> list[np.ndarray]:
    """Create a grid of reference points NSGA-III style

    Args:
        pop (list[Individual]): Population of individuals
        grid_res (int): Resolution of the grid

    Returns:
        list[np.ndarray]: List of reference points
    """
    num_objectives = len(pop[0].fitnessarray)
    ref_points = []

    # Create a grid of reference points, every intiger combination in the size (grid_res, num_objectives) is a point and normalised
    for combination in itertools.combinations_with_replacement(range(grid_res+1), num_objectives):
        if sum(combination) == grid_res:
            point = np.array(combination) / grid_res
            ref_points.append(point)
    
    return ref_points

def associate_reference_points(pop: list[Individual], ref_points: list[np.ndarray]) -> None:
    """Associate the reference points with the individuals

    Args:
        pop (list[Individual]): Population of individuals
        ref_points (list[np.ndarray]): List of reference points
    """
    for ind in pop:
        ind.ref_point = argmin(ref_points, key=lambda x: np.linalg.norm(np.array(ind.fitnessarray) - x))
        ind.ref_dist = float(np.linalg.norm(np.array(ind.fitnessarray) - ref_points[ind.ref_point]))

def niching_selection(fronts, ref_points, pop_size):
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            # Perform niching based on reference points
            selected.extend(select_by_reference(front, ref_points, pop_size - len(selected)))
            break
    return selected

def select_by_reference(pop: list[Individual], ref_points: list[np.ndarray], remaining_spots: int) -> list[Individual]:
    """Apply niching to the population

    Args:
        pop (list[Individual]): Population of individuals
        ref_points (list[np.ndarray]): List of reference points
        remaining_spots (int): Amount of spots left in the population

    Returns:
        list[Individual]: New population selection
    """
    # print(len(pop))
    associate_reference_points(pop, ref_points)
    pop = copy.copy(pop)
    selected = []
    # Select the best individuals for each reference point
    while len(selected) < remaining_spots:
        for ref in range(len(ref_points)):
            individuals = [ind for ind in pop if ind.ref_point == ref]
            if individuals:
                chosen = min(individuals, key=lambda x: (x.ref_dist, -x.crowding_dist))
                selected.append(chosen)
                pop.remove(chosen)
            if len(selected) >= remaining_spots:
                break

    return selected

##### Rest of the owl #####

def mutate(vals: np.ndarray, probability: float) -> np.ndarray:
    """Mutate the values of a layer

    Args:
        vals (np.ndarray): Weights or biases of a layer

    Returns:
        np.ndarray: Mutated weights or biases
    """
    vals += np.where(np.random.uniform(0, 1, size=vals.shape) <= probability, np.random.normal(0, 1, size=vals.shape), 0)

    return vals


# crossover
def crossover(env: Environment, pop: list[Individual], method: tuple[int, int], cfg: dict) -> list[Individual]:
    """
    Perform crossover and mutation on a population of individuals to generate offspring.

    Args:
        env (Environment): The environment in which the individuals are evaluated.
        pop (list[Individual]): The population of individuals to perform crossover on.
        method (int): The method of crossover to use. 
                      0 for snipcombine, other values for blendcombine.
        cfg (dict): Configuration dictionary containing mutation rate and domain limits.

    Returns:
        list[Individual]: A list of offspring individuals generated from the crossover and mutation process.
    """

    # Goes through pairs in the population and chooses two random fit individuals according to tournament
    for _ in range(0,len(pop), 2):
        p1 = tournament(pop, 2)
        p2 = tournament(pop, 2)

        n_offspring =   np.random.randint(1, 6)
        offspring: list[Individual] = []

        for _ in range(n_offspring):
            cross_prop = np.random.uniform(0,1) # Get a random ratio of influence between parents p1 and p2
            child: Individual = Individual()
            child.intialiseWeights(cfg['archetecture']) # Initialize the weights and biases of the child
            wnb: list[tuple[np.ndarray, np.ndarray]] = []
            
            for layer_p1, layer_p2 in zip(p1.wnb, p2.wnb):
                # crossover, we have different methods depending on the island
                if method[0] == 0:
                    weights, bias = snipcombine(layer_p1, layer_p2)
                else:
                    weights, bias = blendcombine(layer_p1, layer_p2, cross_prop)

                # mutation
                if method[1] == 0:
                    weights = mutate(weights, cfg['mutation'])
                    bias = mutate(bias, cfg['mutation'])
                else:
                    weights = swap_mutation(weights, cfg['mutation'])
                    bias = swap_mutation(bias, cfg['mutation'])

                # limit between -1 and 1 using Tanh function
                weights = np.clip(weights, cfg['dom_l'], cfg['dom_u'])
                bias = np.clip(bias, cfg['dom_l'], cfg['dom_u'])

                assert weights.shape == layer_p1[0].shape, "Weights shape does not match"
                assert bias.shape == layer_p1[1].shape, "Biases shape does not match"
                
                wnb.append((weights , bias))
            
            child.setWeights(wnb)
            child.evaluate(env)
            offspring.append(child)
    return offspring

def snipcombine(layer_p1: tuple[np.ndarray, np.ndarray], layer_p2: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Combines two layers by randomly selecting crossover points for weights and biases.
    This function takes two layers, each represented by a tuple of numpy arrays (weights and biases),
    and combines them by selecting random crossover points for the weights and biases. The resulting
    combined weights and biases are returned as a tuple.

    Args:
        layer_p1 (tuple[np.ndarray, np.ndarray]): The first layer's weights and biases.
        layer_p2 (tuple[np.ndarray, np.ndarray]): The second layer's weights and biases.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the combined weights and biases.
    """

    weightspoint = np.random.uniform(0,layer_p1[0].shape[0])
    biaspoint = np.random.uniform(0,layer_p1[1].shape[0])
    weights = np.concatenate((layer_p1[0][:int(weightspoint)], layer_p2[0][int(weightspoint):]))
    bias = np.concatenate((layer_p1[1][:int(biaspoint)], layer_p2[1][int(biaspoint):]))
    assert weights.shape == layer_p1[0].shape, "Weights shape does not match"
    assert bias.shape == layer_p1[1].shape, "Biases shape does not match"
    return weights,bias

def blendcombine(layer_p1: tuple[np.ndarray, np.ndarray], layer_p2: tuple[np.ndarray, np.ndarray], cross_prob: float):
    """
    Combines two layers' weights and biases using a blend crossover method.
    Args:
        layer_p1 (tuple[np.ndarray, np.ndarray]): The first layer's weights and biases.
        layer_p2 (tuple[np.ndarray, np.ndarray]): The second layer's weights and biases.
        cross_prob (float): The crossover probability, determining the blend ratio.

    Returns:
        tuple: A tuple containing the combined weights and biases.
    """

    weights = layer_p1[0]*cross_prob+layer_p2[0]*(1-cross_prob)
    bias = layer_p1[1]*cross_prob+layer_p2[1]*(1-cross_prob)
    assert weights.shape == layer_p1[0].shape, "Weights shape does not match"
    assert bias.shape == layer_p1[1].shape, "Biases shape does not match"
    return weights, bias

def doomsday(env: Environment, pop: list[Individual], cfg: dict) -> list[Individual]:
    """Kills the worst genomes, and replace with new best/random solutions

    Args:
        env (Environment): Environment object for the evoman framework
        pop (list[Individual]): Population formatted as a an array of models containing tuples of weights and biases.
        cfg (dict): Configuration dictionary

    Returns:
        list[Individual]: New population
    """

    amount_worst = int(len(pop) * 0.75)
    # Use pareto front sorting to get the worst individuals
    worst_indices = argsort(pop, key=lambda x: (x.front, -x.crowding_dist))
    worst_indices = worst_indices[amount_worst:]
    best_individual = min(pop, key=lambda x: (x.front, -x.crowding_dist))

    # Nuke the hell out of the worst, make them mutate like crazy
    for idx in worst_indices:
        wnb = []
        for layer_idx, (weights, biases) in enumerate(pop[idx].wnb):
            prob = np.random.uniform(0, 1)
            if prob <= 0.8:
                weights, biases = nuke(cfg, weights, biases)
            else:
                weights = mutate(best_individual.wnb[layer_idx][0], cfg['mutation'])
                biases = mutate(best_individual.wnb[layer_idx][1], cfg['mutation'])
            wnb.append((weights, biases))

        pop[idx].setWeights(wnb)
        pop[idx].evaluate(env)

    return pop

def nuke(cfg: dict, weights: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nuke the weights and biases of a layer

    Args:
        cfg (dict): Configuration dictionary
        weights (np.ndarray): Weights of a layer
        biases (np.ndarray): Biases of a layer

    Returns:
        tuple[np.ndarray, np.ndarray]: New weights and biases
    """
    # Create mutation mask for weights
    mutation_mask_w = np.random.uniform(0, 1, size=weights.shape) <= np.random.uniform(0, 1)
    weights = np.where(mutation_mask_w, np.random.uniform(cfg['dom_l'], cfg['dom_u'], size=weights.shape), weights)

    # Create mutation mask for biases
    mutation_mask_b = np.random.uniform(0, 1, size=biases.shape) <= np.random.uniform(0, 1)
    biases = np.where(mutation_mask_b, np.random.uniform(cfg['dom_l'], cfg['dom_u'], size=biases.shape), biases)
    
    return weights, biases

def generate_new_pop(envs: list[Environment], npop: int, n_hidden_neurons: list[int]) -> tuple[list[Individual], tuple[int, float, float], int]:
    """Generate a new population of individuals

    Args:
        envs (list[Environment]): List of Environment objects for the evoman framework
        npop (int): Amount of individuals in the population
        n_hidden_neurons (list[int]): Amount of hidden neurons in each layer

    Returns:
        tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, tuple[int, float, float], int]: Population, fitness values, best individual, mean and standard deviation of the fitness values, and the initial generation number
    """
    pop: list[Individual] = []
    for i in range(npop):
        individual = Individual(i)
        individual.intialiseWeights(n_hidden_neurons)
        individual.evaluate(envs[0])
        pop.append(individual)

    best = int(argmax(pop, key=lambda x: x.fitness))
    mean = sum(pop) / float(len(pop))
    std = stdev(pop)
    ini_g = int(0)
    env.update_solutions(pop)
    return pop, (best, mean, std), ini_g

# def load_pop(env: Environment, experiment_name: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, tuple[int, float, float], int]:
#     """Load a population from a previous experiment

#     Args:
#         env (Environment): Environment object for the evoman framework
#         experiment_name (str): Name of the experiment

#     Returns:
#         tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, tuple[int, float, float], int]: Population, fitness values, best individual, mean and standard deviation of the fitness values, and the initial generation number
#     """
#     env.load_state()
#     pop = env.solutions[0]
#     fit_pop = env.solutions[1]

#     best = int(np.argmax(fit_pop))
#     mean = float(np.mean(fit_pop))
#     std = float(np.std(fit_pop))

#     # finds last generation number
#     file_aux  = open(experiment_name+'/gen.txt','r')
#     ini_g = int(file_aux.readline())
#     file_aux.close()
#     return pop, fit_pop, (best, mean, std), ini_g

def make_species(pop: list[Individual], cfg: dict) -> list[list[Individual]]:
    """Split the population into species

    Args:
        pop (list[Individual]): Population of individuals
        cfg (dict): Configuration dictionary

    Returns:
        list[list[Individual]]: Species populations, fitness values, and other information
    """
    species_pop: list[list[Individual]] = []
    spec_size = cfg['npop'] // cfg['species']
    for i in range(cfg['species']):
        species_pop.append(pop[i*spec_size:(i+1)*spec_size])
    
    return species_pop

def cross_species(species_pop: list[list[Individual]], cfg: dict) -> list[list[Individual]]:
    """Crossover between species

    Args:
        species_pop (list[list[Individual]]): Species population
        cfg (dict): Configuration dictionary

    Returns:
        list[list[Individual]]: New species population
    """
    prob = random.uniform(0,1)
    
    if cfg['spec_cross'] >= prob:
        print("Crossing species")
        best_specs = [copy.copy(max(spec)) for spec in species_pop]
        print(f"Best species: {best_specs}")
        
        for target_island in range(cfg['species']):
            origin_island = (target_island + 1) % cfg['species']
            print(f"Origin island: {origin_island} {best_specs[origin_island]}, Target island: {species_pop[target_island][argmax(species_pop[target_island])]}")
            species_pop[target_island][argmax(species_pop[target_island])] = best_specs[origin_island]
        print(f"Best species after crossing: {[max(spec) for spec in species_pop]}\n")
        return species_pop
    return species_pop

def train(envs: list[Environment], pop: list[Individual], best: int, ini_g: int, cfg: dict) -> None:
    """Train/Evolution loop for the genetic algorithm

    Args:
        env (Environment): Environment object for the evoman framework
        pop (list[Individual]): Population formatted as a an array of Individuals containing metrics, weights and biases.
        best (int): Index of the best individual in the population
        ini_g (int): Which generation to start at
        cfg (dict): Configuration dictionary
    """
 
    no_improvement = 0
    last_best: float = pop[best].fitness
    cur_env = 0

    species_pop = make_species(pop, cfg)
    species_ref_points = [make_reference_points(spec, cfg['grid_res'] + i) for i, spec in enumerate(species_pop)]
    spec_notimproved = [0]*cfg['species']

    # Just extra keeping track of the best solution, as not to lose this
    best_individuals = []
    for specimen in species_pop:
        best_index = argmax(specimen, key=lambda x: x.fitness)
        best_individuals.append(pop[best_index])

    # Main training loop
    for g in range(ini_g+1, cfg['gens']):
        # Loop through all the islands and train them for one generation
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print(f"Best individuals begin: {[max(spec) for spec in species_pop]}")
        for i, island_pop in enumerate(species_pop):
            species_pop[i], spec_notimproved[i], best_individuals[i] = train_spec(envs[cur_env], island_pop, best_individuals[i], spec_notimproved[i], (cfg['comb_meths'][i], cfg['muttypes'][i]), cfg['scaletype'][i], species_ref_points[i], cfg) # type: ignore
        print(f"Best individuals mid  : {[max(spec) for spec in species_pop]}")
        # Make lists of the complete population to log the absolute best individual and other statistics
        total_pop: list[Individual] = [p for spec in species_pop for p in spec]

        # Get statistics
        best = argmax(total_pop, key=lambda x: x.fitness)
        std  = stdev(total_pop, key=lambda x: x.fitness)
        mean = sum(total_pop) / float(len(total_pop))

        # Memory for best solution
        best_individual = max(total_pop)

        # saves results
        save_txt(experiment_name+'/results.txt', '\n'+str(g)+' '+str(round(best_individual.fitness,6))+' '+str(round(mean,6))+' '+str(round(std,6)), 'a')
        print( '\n GENERATION '+str(g)+' '+str(round(best_individual.fitness,6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        wandb.log({'Generation': ini_g, 'Best Fitness': best_individual.fitness, 'Mean Fitness': mean, 'Std Fitness': std, 'Best Player Health': best_individual.player_energy, 'Best Enemy Health': best_individual.enemy_energy, 'Best Timesteps': best_individual.timesteps,'Gain': best_individual.gain, 'Best Dead Enemies': best_individual.dead_enemies, 'Current Env': cur_env})

        # saves generation number
        save_txt(experiment_name+'/gen.txt', str(i))

        # saves file with the best solution
        save_weights(experiment_name, best_individual.wnb)

        # saves simulation state
        env = envs[cur_env]
        # env.update_solutions(total_pop)
        # env.save_state()

        species_pop = cross_species(species_pop, cfg)
        cur_best = max(total_pop).fitness
        if cur_best <= last_best:
            no_improvement += 1
        else:
            last_best = cur_best
            no_improvement = 0
        
        assert best_individual >= cur_best, f"Best individual {best_individual} is not the same as the best in the population {cur_best}. {max(total_pop)}"
        assert best_individual in [p for spec in species_pop for p in spec], f"Best individual {best_individual} is not in the population"

        # Go to the next environment with more enemies if no improvement or high fitness
        # if total_pop[best] > 90 or no_improvement >= 450:
        #     cur_env += 1
        #     cur_env = min(cur_env, len(envs)-1) # Make sure we don't go out of bounds with the environments

        #     # Re-evaluate the population in the new environment
        #     for best_ind in best_individuals:
        #         best_ind.evaluate(envs[cur_env])
            
        #     for spec in species_pop:
        #         for p in spec:
        #             p.evaluate(envs[cur_env])

        #     no_improvement = 0

        # Check if the best stored individual is still the best in the population (or one that's better)
        # appendbestind = True
        # for p in species_pop:
        #     if p[argmax(p)] >= best_individual:
        #         appendbestind = False

        # if appendbestind:
        #     species_select = np.random.randint(0, len(species_pop))
        #     species_pop[species_select][argmin(species_pop[species_select])] = best_individual

        print(f"Best individuals end  : {[max(spec) for spec in species_pop]}")


    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')

    file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

    env.state_to_log() 

def train_spec(env: Environment, 
               pop: list[Individual],
               best_individual: Individual, 
               spec_notimproved: int, 
               comb_meth: tuple[int,int], 
               scale_type: int,
               ref_points: list[np.ndarray],
               cfg: dict) -> tuple[list[Individual], float, Individual]:
    """Train/Evolution loop for the genetic algorithm, for one island of species

    Args:
        env (Environment): Environment object for the evoman framework
        pop (list[Individual]): Population formatted as a an array of Individuals containing tuples of weights and biases.
        best_individual (Individual): Best individual in the population of all generations (the best agent is invincible?!)
        spec_notimproved (int): Amount of generations the species has not improved
        comb_meth (int): Crossover method
        scale_type (int): Scaling type
        cfg (dict): Configuration dictionary

    Returns:
        tuple[list[Individual], float, Individual]: Population, amount of generations the species has not improved, best individual in the population
    """
    # beginBest = copy.copy(max(pop))

    # Pareto front sorting
    fronts = non_dominant_sorting(pop)
    for front in fronts:
        crowding_distance_sorting(front)

    offspring = crossover(env, pop, comb_meth, cfg)  # crossover
    # Add offspring to the population
    pop = pop + offspring
    # if max(pop) < beginBest:
        # print(f"Best individual worsened: {max(pop)} {beginBest}. Crossover")
    
    fronts = non_dominant_sorting(pop)
    for front in fronts:
        crowding_distance_sorting(front)

    # selection
    pop = selection(env, pop, scale_type, ref_points, cfg)
    # if max(pop) < beginBest:
        # print(f"Best individual worsened: {max(pop)} {beginBest}. Selection")

    # searching new areas
    best_sol = min(pop, key=lambda x: (x.front, -x.crowding_dist))
    # if max(pop) < beginBest:
        # print(f"Best individual worsened: {best_sol} {beginBest}. Evaluation")

    # Update so that the fitness of the stored best individual is still correct in the current environment
    if best_sol <= best_individual:
        spec_notimproved += 1
    else:
        best_individual = min(pop, key=lambda x: (x.front, -x.crowding_dist))
        spec_notimproved = 0
    
    if spec_notimproved >= cfg['doomsteps']:
        save_txt(experiment_name+'/results.txt', '\ndoomsday', 'a')
        # assert max(pop) == best_sol, f"Best fit {best_sol} not in population 3 {max(pop)}"
        pop = doomsday(env, pop, cfg)
        spec_notimproved = 0
    # if min(pop, key=lambda x: (x.front, -x.crowding_dist)) < beginBest:
        # print(f"Best individual worsened: {max(pop)} {beginBest}. Doomsday")
    # Replace one in population with the best stored individual for this island
    # if best_individual > max(pop):
    #     replace_index = argmin(pop)
    #     pop[replace_index] = copy.copy(best_individual)

    # Check if all id's are unique, if not, change them
    ids = [p.id for p in pop]
    if len(ids) != len(set(ids)):
        for i, p in enumerate(pop):
            p.id = i
            
    return pop, spec_notimproved, best_individual

def selection(env: Environment, pop: list[Individual], scale_type: int, ref_points: list[np.ndarray], cfg: dict): # This got hella ugly and messy
    best = argmin(pop, key=lambda x: (x.front, -x.crowding_dist)) #best solution in generation
    pop_cp = pop.copy()
    bestFits = []
    for i in range(2):
        pop[best].evaluate(env) # repeats best eval, for stability issues
        bestFits.append(pop[best].fitness)
    # Check if all in bestFits are the same
    if all([f != bestFits[0] for f in bestFits]):
        print(f"Best fitnesses are not the same: {bestFits}")
    best = argmax(pop) #best solution in generation
    fit2scale = np.array(list(map(lambda x: x.fitness, pop_cp))) # Get list of fitnesses to scale

    if scale_type == 1: # Defined in config file
        fit_pop_norm = sigma_scaling(fit2scale, 2)
    elif scale_type == 0:
        fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit2scale), fit2scale))) # avoiding negative probabilities, as fitness is ranges from negative numbers
    elif scale_type == 2:
        chosen = argsort(pop_cp, key=lambda x: (x.front, -x.crowding_dist))[:cfg['npop'] // cfg['species']]
    elif scale_type == 3:
        pop = niching_selection(non_dominant_sorting(pop_cp), ref_points, cfg['npop'] // cfg['species'])


    if scale_type in [0,1]:
        probs = (fit_pop_norm)/(fit_pop_norm).sum() # normalize fitness values to probabilities
        # Chose indices corresponding to individuals in the population, randomly chosen according to their fitness
        chosen = np.random.choice(len(pop), cfg['npop'] // cfg['species'] , p=probs, replace=False)
        chosen = np.append(chosen[1:],best) # Just to be sure we don't lose the best individual

    # Replace the population with the chosen individuals
    if scale_type in [0,1,2]:
        pop2replace: list[Individual] = [pop[int(c)] for c in chosen]
        pop = pop2replace

    assert len(pop) == cfg['npop'] // cfg['species'], f"Population size is not correct: {len(pop)}"
    return pop


#### UTILS ####


# normalizes
def norm(x, pfit_pop):
    
    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm

def print_dict(d: dict) -> None:
    """Print a dictionary

    Args:
        d (dict): Dictionary to print
    """
    for k, v in d.items():
        print(f'{k}: {v}')

def stdev(lst, key=lambda x: x) -> float:
    """Calculate the standard deviation of a list with a custom key function

    Args:
        lst (list): List of values
        key (function): Function to extract a comparison key from each element

    Returns:
        float: Standard deviation of the list
    """
    values = [key(x) for x in lst]
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return sqrt(variance)

# For explanation of arg functions look at the corresponding numpy versions (np.argmax etc.)
def argmax(iterable, key=lambda x: x): # adapted from https://stackoverflow.com/questions/16945518/finding-the-index-of-the-value-which-is-the-min-or-max-in-python
    return min(enumerate(iterable), key=lambda x: key(x[1]))[0]

def argmin(iterable, key=lambda x: x):
    return min(enumerate(iterable), key=lambda x: key(x[1]))[0]

def argsort(seq, key=lambda x: x): 
    return sorted(range(len(seq)), key=lambda i: key(seq[i]))

def save_weights(name: str, individual: list[tuple[np.ndarray, np.ndarray]]) -> None:
    file = gzip.open(name+'/best', 'wb', compresslevel = 5)
    pickle.dump(individual, file, protocol=2) # type: ignore
    file.close()

def save_txt(name: str, text: str, savetype: str = 'w') -> None:
    file_aux  = open(name,savetype)
    file_aux.write(text)
    file_aux.close()



def main(envs: list[Environment], args: argparse.Namespace, cfg: dict) -> None:
    """Main function for the genetic algorithm

    Args:
        env (Environment): Environment object for the evoman framework
        args (argparse.Namespace): Command line arguments
        cfg (dict): Configuration dictionary
    """
    # loads file with the best solution for testing
    if args.run_mode =='test':
        file = gzip.open(args.experiment_name+'/best')
        bsol =  pickle.load(file, encoding='latin1')
        ind = Individual()
        ind.setWeights(bsol, force=True)
        print( '\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed','normal')
        ind.evaluate(env)
        sys.exit(0)


    # initializes population loading old solutions or generating new ones
    if not os.path.exists(args.experiment_name+'/evoman_solstate') or args.new_evolution:
        print( '\nNEW EVOLUTION\n')
        pop, fit_pop_stats, ini_g = generate_new_pop(envs, cfg['npop'], cfg['archetecture'])

    # else:
    #     print( '\nCONTINUING EVOLUTION\n')
    #     pop, fit_pop, fit_pop_stats, ini_g = load_pop(env, args.experiment_name)

    # saves results for first pop
    best, mean, std = fit_pop_stats
    save_txt(experiment_name+'/results.txt', '\n\ngen best mean std' + '\n'+str(ini_g)+' '+str(round(pop[best].fitness,6))+' '+str(round(mean,6))+' '+str(round(std,6)), 'a')
    print( '\n GENERATION '+str(ini_g)+' '+str(round(pop[best].fitness ,6))+' '+str(round(mean,6))+' '+str(round(std,6)))

    train(envs, pop, best, ini_g, cfg)


if __name__ == "__main__": # Basically just checks if the script is being run directly or imported as a module

    # Command line arguments, this makes it so you can run the script from the command line with different arguments as such:
    # python optimization_specialist_demo.py --experiment_name=optimization_test --headless=True --new_evolution=False --run_mode=train
    parser = argparse.ArgumentParser(description='Run the genetic algorithm for the evoman framework')
    parser.add_argument('--experiment_name', type=str, default='optimization_test', help='Name of the experiment')
    parser.add_argument('--headless', type=bool, default=True, help='Run the simulation without visuals')
    parser.add_argument('--new_evolution', type=bool, default=False, help='Start a new evolution')
    parser.add_argument('--run_mode', type=str, default='train', help='Run mode for the genetic algorithm')
    parser.add_argument('--wandb', action='store_true', help='Use Weights and Biases for logging')
    args = parser.parse_args()

    # Loading our training config
    cfg: dict = yaml.safe_load(open('./config/' + args.experiment_name + '.yaml')) # type: ignore
    
    print('Config:')
    print_dict(cfg)

    # Initialize Weights and Biases
    wandb.init(
        # set the wandb project where this run will be logged
        project="Evoman Competition",

        # track hyperparameters and run metadata
        name=args.experiment_name,
        config=cfg
    )

    # choose this for not using visuals and thus making experiments faster
    headless = args.headless
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy" # Turn off videodriver when running headless

    # Create a folder to store the experiment
    experiment_name = args.experiment_name
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initializes simulation in individual evolution mode, for single static enemy.
    envs = []
    for i in range(len(cfg['enemies'])):
        if not os.path.exists(experiment_name+f'{i}'):
            os.makedirs(experiment_name+f'{i}')
        env = Environment(experiment_name=experiment_name+f'{i}',
                        enemies=cfg['enemies'][i],
                        multiplemode= 'yes' if len(cfg['enemies'][i]) > 1 else 'no',
                        playermode="ai",
                        player_controller=player_controller(cfg['archetecture']), # Initialise player with specified archetecture
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False,
                        clockprec='medium')
        env.state_to_log() # checks environment state
        envs.append(env)


    ####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
    ini = time.time()  # sets time marker

    main(envs, args, cfg)