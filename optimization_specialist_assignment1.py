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

# imports other libs
import time
import numpy as np
import random
from math import fabs, sqrt
import glob, os
import gzip, pickle, yaml
import argparse
import math
import wandb

from multiprocessing import Pool

# runs simulation
def simulation(env, x) -> tuple[float, tuple[float, float, float]]:
    # use pcont(bias and weights of nn)as input, return fitness
    f, p, e, t = env.play(pcont=x)
    return f, (p, e, t)


# normalizes x to be between 0 and 1
def norm(x, pfit_pop):
    if (max(pfit_pop) - min(pfit_pop)) > 0:
        x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


# evaluation generation x's fitness
def evaluate(x: list[list[tuple[np.ndarray, np.ndarray]]]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the fitness of a population of individuals

    Args:
        x (list[list[tuple[np.ndarray, np.ndarray]]]): Population of individuals, each individual is a list of tuples containing weights and biases for each layer

    Returns:
        np.ndarray: List of fitness values for each individual in the population of size (pop_size*1)
    """
    fitness, extra_info = zip(*list(map(lambda y: simulation(env, y), x)))
    return np.array(fitness), np.array(extra_info)


'''
# clear version for upper function
def evaluate(population):
    """
    Evaluates the fitness of a population of individuals.

    Args:
        population: A list of individuals. Each individual is a set of weights and biases for a neural network.

    Returns:
        A list of fitness values, where each individual has one corresponding fitness value.
    """
    fitness_values = []

    
    for individual in population:    
        fitness = simulation(env, individual)
        fitness_values.append(fitness)

    return np.array(fitness_values)
'''


# tournament
def tournament(pop: list[list[tuple[np.ndarray, np.ndarray]]], fit_pop: np.ndarray) -> tuple[
    list[tuple[np.ndarray, np.ndarray]], int]:
    """Tournament function to select the fittest of two individuals in the population

    Args:
        pop (list[list[tuple[np.ndarray, np.ndarray]]]): Population of shape (pop_size*layer_amt*2)
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)

    Returns:
        list(tuple(weigths, bias)): One individual in the population
    """

    # Get two random values that correspond to two individuals in the population
    c1 = random.randint(0, len(pop) - 1)
    c2 = random.randint(0, len(pop) - 1)

    # Return the fittest of the two 
    if fit_pop[c1] > fit_pop[c2]:
        return pop[c1], c1
    else:
        return pop[c2], c2


'''
# clear version for upper function
def tournament(population, fitness_values):
    """
    Randomly selects two individuals from the population and returns the one with higher fitness.

    Args:
        population: A list of individuals
        fitness_values: A list of fitness values

    Returns:
        A tuple containing the fittest individual and its index.
    """
    # Randomly pick two different individuals
    index1, index2 = random.sample(range(len(population)), 2)

    # Compare their fitness values, and select the one with the higher fitness
    if fitness_values[index1] > fitness_values[index2]:
        winner = population[index1]
        winner_index = index1
    else:
        winner = population[index2]
        winner_index = index2

    # Return the winning individual and its index
    return winner, winner_index
'''


# Mutation
def standard_mutate(vals: np.ndarray, probability: float) -> np.ndarray:
    """Mutate the values of a layer

    Args:
        vals (np.ndarray): Weights or biases of a layer

    Returns:
        np.ndarray: Mutated weights or biases
    """

    for i in range(0, len(vals)):
        if np.random.uniform(0, 1) <= probability:
            vals[i] = vals[i] + np.random.uniform(0, 1)

    return vals


def nonlinear_mutation(vals: np.ndarray, probability: float) -> np.ndarray:
    """
    Non-linear mutation that applies a non-linear function (e.g., tanh) to the mutation.

    Args:
        vals (np.ndarray): Weights or biases of a layer
        probability (float): Probability of mutating each value

    Returns:
        np.ndarray: Mutated weights or biases
    """
    for i in range(len(vals)):
        if np.random.uniform(0, 1) <= probability:
            vals[i] = vals[i] + np.tanh(np.random.normal(0, 1))

    return vals


def non_uniform_mutation(vals: np.ndarray, probability: float, current_gen: int, max_gen: int) -> np.ndarray:
    """
    Non-uniform mutation where the mutation intensity decreases as the generations progress.

    Args:
        vals (np.ndarray): Weights or biases of a layer
        probability (float): Probability of mutating each value
        current_gen (int): The current generation number
        max_gen (int): The maximum number of generations

    Returns:
        np.ndarray: Mutated weights or biases
    """
    progress = min(current_gen / max_gen, 0.99)

    for i in range(len(vals)):
        if np.random.uniform(0, 1) <= probability:
            # the delta value(mutation intensity) decreases as the generations progress
            delta = (1 - progress) ** 2
            vals[i] += np.random.uniform(-delta, delta)
    return vals


def swap_mutation(vals: np.ndarray, probability: float) -> np.ndarray:
    """
    randomly selects two elements in vals and swaps them.

    Args:
        vals (np.ndarray): Weights or biases of a layer
        probability (float): Probability of performing a swap mutation

    Returns:
        np.ndarray: Mutated weights or biases
    """
    if np.random.uniform(0, 1) <= probability:
        idx1, idx2 = np.random.choice(len(vals), size=2, replace=False)  # Randomly select two distinct indices
        # Swap the values at the two indices
        vals[idx1], vals[idx2] = vals[idx2], vals[idx1]

    return vals


def parent_similarity(parent1, parent2, threshold=1.):
    """
    Checks if two parents are too similar based on the Euclidean distance.

    Args:
        parent1, parent2: Two parent individuals.
        threshold: Once the distance is below this threshold, the parents are considered too similar.

    Returns:
        bool: True if the parents are too similar, otherwise False.
    """
    distance = 0
    # Calculate the Euclidean distance by iterating through each layer's weights
    for (weights1, biases1), (weights2, biases2) in zip(parent1, parent2):

        distance += np.linalg.norm(weights1 - weights2)

        # Calculate the Euclidean distance between the biases
        distance += np.linalg.norm(biases1 - biases2)

    # Return True if the total distance is less than the threshold, meaning they are too similar
    return distance < threshold


# Crossover
def crossover(pop: list[list[tuple[np.ndarray, np.ndarray]]],
              fit_pop: np.ndarray,
              gen: int,
              cfg: dict,
              n_parent: int = 5) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Crossover function to generate offspring from the population

    Args:
        pop (list[tuple[np.ndarray, np.ndarray]]): Population formatted as a an array of models containing tuples of weights and biases.
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)
        cfg (dict): Configuration dictionary
        n_parent (int, optional): Amount of parents to select for each offspring

    Returns:
        np.ndarray: New population of offspring
    """

    pop_copy = pop.copy()
    fit_pop_copy = fit_pop.copy()  # Fit pop cop with a hat on top
    offspring: list[list[tuple[np.ndarray, np.ndarray]]] = []
    # Know that the 1st of pop_copy corresponds to the 1st of fit_pop_copy

    # Control the batch nums for crossover
    for p in range(0, len(pop), n_parent):
        parents = []
        ogparents = []
        parent_shapes = []

        # Select `parent_amt` parents using tournament selection
        for _ in range(n_parent):
            parentSnips = []
            parent, del_fit = tournament(pop_copy, fit_pop)  # del_fit is the index of the parent that was selected
            shape = []

            # Incest prevention
            # if len(ogparents) > 0:
            #     while any(parent_similarity(parent, existing_parent) for existing_parent in ogparents):
            #         parent, del_fit = tournament(pop_copy, fit_pop)

            ogparents.append(parent)

            # Remove the parent from the population
            pop_copy.pop(del_fit)
            fit_pop_copy = np.delete(fit_pop_copy, del_fit)

            for layer in parent:
                # Go through every layer in parents
                # Layer_p1 looks like tuple(np.ndarray(weights), np.ndarray(bias))
                # weight [(1,2,*3,4,*5,6,*7,8,*9,10,1,2,*3,4,*5,6,*7,8,*9,10,1,2,*3,4,*5,6,*7,8,*9,10)]
                # weight 4 waardes (die niet hetzelfde zijn) tussen 0 en 9
                # crossover
                snips = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

                weights = layer[0]
                bias = layer[1]

                # Save the shape of the weights and bias
                shape.append((weights.shape, bias.shape))
                # Reshape to one dimensional array
                weights = weights.reshape((weights.shape[0] * weights.shape[1]))
                bias = bias.reshape((bias.shape[0] * bias.shape[1]))

                weightSnips = []
                biasSnips = []
                for snipInd in range(len(snips) - 1):
                    startSnip = snips[snipInd]
                    endSnip = snips[snipInd + 1]

                    subtract = 0
                    if snipInd == len(snips) - 2:
                        subtract -= 1

                    wSnip = getSnippet(weights, startSnip, endSnip, subtract)
                    weightSnips.append(wSnip)

                    bSnip = getSnippet(bias, startSnip, endSnip, subtract)
                    biasSnips.append(bSnip)

                parentSnips.append((weightSnips, biasSnips))

            parents.append(parentSnips)
            parent_shapes.append(shape)

        n_offspring = 5

    for offset in range(0, n_offspring):
        child: list[tuple[np.ndarray, np.ndarray]] = []

        # Go through each layer of the parents
        for layerNum in range(len(parents[0])):
            weights = np.array(0)
            bias = np.array(0)

            for snip in range(len(snips) - 1):
                snippetW = parents[(snip + offset) % 5][layerNum][0][snip]
                snippetB = parents[(snip + offset) % 5][layerNum][1][snip]
                weights = np.append(weights, snippetW)
                bias = np.append(bias, snippetB)

            # mutation
            weights = weights.reshape(parent_shapes[0][layerNum][0])
            bias = bias.reshape(parent_shapes[0][layerNum][1])

            if cfg['muttype'] == 'standard':
                weights = standard_mutate(weights, cfg['mutation'])
                bias = standard_mutate(bias, cfg['mutation'])
            elif cfg['muttype'] == 'nonlinear':
                weights = nonlinear_mutation(weights, cfg['mutation'])
                bias = nonlinear_mutation(bias, cfg['mutation'])
            elif cfg['muttype'] == 'non_uniform':
                weights = non_uniform_mutation(weights, cfg['mutation'], gen, cfg['gens'])
                bias = non_uniform_mutation(bias, cfg['mutation'], gen, cfg['gens'])

            # limit between -1 and 1
            bias = bias.clip(cfg['dom_l'], cfg['dom_u'])
            weights = weights.clip(cfg['dom_l'], cfg['dom_u'])

            layerChild = (weights, bias)
            child.append(layerChild)
        offspring.append(child)

    return offspring


def getSnippet(vals, startSnip, endSnip, subtract):
    startW = math.floor((vals.shape[0] - 1) * startSnip) - subtract  # Begin index van snippet
    endW = math.floor((vals.shape[0] - 1) * endSnip) - subtract  # End index van snippet
    wSnip = vals[startW:endW]
    return wSnip


def doomsday(pop: list[list[tuple[np.ndarray, np.ndarray]]], fit_pop: np.ndarray, info_pop: np.ndarray, cfg: dict) -> tuple[
    list[list[tuple[np.ndarray, np.ndarray]]], np.ndarray]:
    """Kills the worst genomes, and replace with new best/random solutions

    Args:
        pop (list[tuple[np.ndarray, np.ndarray]]): Population formatted as a an array of models containing tuples of weights and biases.
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)

    Returns:
        tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]: New population and fitness values
    """

    worst = int(cfg['npop'] // 4)  # a quarter of the population
    order = np.argsort(fit_pop)  # sort the population by fitness
    orderasc = order[0:worst]  # get the worst individuals

    for o in orderasc:
        for l, layer in enumerate(pop[o]):
            for v, vect in enumerate(layer):  # Go through the bias and weights vectors
                for i in range(0, len(vect)):
                    pro = np.random.uniform(0, 1)  # Get a random probability
                    if np.random.uniform(0, 1) <= pro:
                        pop[o][l][v][i] = np.random.uniform(cfg['dom_l'], cfg['dom_u'])  # random dna, uniform dist.
                    else:
                        pop[o][l][v][i] = pop[order[-1:][0]][l][v][
                            i]  # dna from best, which is the last index (-1) of the order list

        val, extra_info = evaluate([pop[o]])  # Evaluate the new individual
        info_pop[o] = extra_info[0]  # Update the fitness value
        fit_pop[o] = val[0]

    return pop, fit_pop


def generate_new_pop(npop: int, n_hidden_neurons: list[int]) -> tuple[
    list[list[tuple[np.ndarray, np.ndarray]]], np.ndarray, tuple[int, float, float], int]:
    """Generate a new population of individuals

    Args:
        npop (int): Amount of individuals in the population
        n_hidden_neurons (list[int]): Amount of hidden neurons in each layer

    Returns:
        tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, tuple[int, float, float], int]: Population, fitness values, best individual, mean and standard deviation of the fitness values, and the initial generation number
    """
    pop: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for i in range(npop):
        individual = []
        in_size = 20  # Amount of input neurons (sensors of the game)
        for layer_size in n_hidden_neurons:
            weights = np.random.uniform(-1, 1, (in_size, layer_size))
            bias = np.random.uniform(-1, 1, (1, layer_size))
            in_size = layer_size
            individual.append((weights, bias))
        pop.append(individual)

    ###############################################################################

    # [
    #     [                                      Model 1
    #         ([[1,1,...,0.5],...,[1,3,...,0.2]], [1,2,...,0.1])     Linear layer 1 (weights, bias)
    #     ],
    # ]

    ###############################################################################

    fit_pop, extra_info = evaluate(pop)
    best = int(np.argmax(fit_pop))
    mean = float(np.mean(fit_pop))
    std = float(np.std(fit_pop))
    ini_g = int(0)
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    return pop, fit_pop, extra_info, (best, mean, std), ini_g


def load_pop(env: Environment, experiment_name: str) -> tuple[
    list[tuple[np.ndarray, np.ndarray]], np.ndarray, tuple[int, float, float], int]:
    """Load a population from a previous experiment

    Args:
        env (Environment): Environment object for the evoman framework
        experiment_name (str): Name of the experiment

    Returns:
        tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, tuple[int, float, float], int]: Population, fitness values, best individual, mean and standard deviation of the fitness values, and the initial generation number
    """
    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = int(np.argmax(fit_pop))
    mean = float(np.mean(fit_pop))
    std = float(np.std(fit_pop))

    # finds last generation number
    file_aux = open(experiment_name + '/gen.txt', 'r')
    ini_g = int(file_aux.readline())
    file_aux.close()
    return pop, fit_pop, (best, mean, std), ini_g


def train(env: Environment, pop: list[list[tuple[np.ndarray, np.ndarray]]], fit_pop: np.ndarray, info_pop: np.ndarray, best: int, ini_g: int,
          cfg: dict) -> None:
    """Train/Evolution loop for the genetic algorithm

    Args:
        env (Environment): Environment object for the evoman framework
        pop (list[list[tuple[np.ndarray, np.ndarray]]]): Population formatted as a an array of models containing tuples of weights and biases.
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)
        best (int): Index of the best individual in the population
        ini_g (int): Which generation to start at
        cfg (dict): Configuration dictionary
    """

    last_sol = fit_pop[best]
    doomsday_counter = 0

    # cfg['gens'] is the max number of generations
    for i in range(ini_g + 1, cfg['gens']):

        # generation new offspring using crossover
        offspring = crossover(pop, fit_pop, i, cfg)  # crossover
        fit_offspring, info_offspring = evaluate(offspring)  # evaluation

        # add offspring to current population
        pop = pop + offspring
        fit_pop = np.append(fit_pop, fit_offspring)
        info_pop = np.append(info_pop, info_offspring, axis=0)

        # Get the best solution
        best = int(np.argmax(fit_pop))  # best solution in generation
        best_fit, best_info = evaluate([pop[best]])  # repeats best eval, for stability issues
        fit_pop[best] = float(best_fit[0])
        info_pop[best] = best_info[0]
        best_sol = fit_pop[best]

        # selection
        fit_pop_cp = fit_pop.copy()
        #fit_pop_norm = np.array(list(map(lambda y: norm(y, fit_pop_cp),fit_pop)))  # avoiding negative probabilities, as fitness is ranges from negative numbers
        fit_pop_scaled = sigma_scaling(fit_pop)
        # Calculate the probabilities of each individual being selected
        probs = (fit_pop_scaled) / (fit_pop_scaled).sum()  # normalize fitness values to probabilities
        # Select the individuals of next generation based on the normed probabilities
        chosen = np.random.choice(len(pop), cfg['npop'], p=probs, replace=False)
        # Make sure the best individual is in the next generation
        chosen = np.append(chosen[1:], best)

        # Create the new population
        newpop: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for c in chosen:
            newpop.append(pop[c])
        pop = newpop

        newfit = []
        for c in chosen:
            newfit.append(fit_pop[c])
        fit_pop = np.array(newfit)

        # searching new areas

        # doomsday trigger
        if best_sol <= last_sol:
            doomsday_counter += 1
        else:
            last_sol = best_sol
            doomsday_counter = 0

        if doomsday_counter >= 15:
            file_aux = open(experiment_name + '/results.txt', 'a')
            file_aux.write('\ndoomsday')
            file_aux.close()

            pop, fit_pop = doomsday(pop, fit_pop, info_pop, cfg)
            doomsday_counter = 0

        best = int(np.argmax(fit_pop))
        std = np.std(fit_pop)
        mean = np.mean(fit_pop)

        # saves results
        file_aux = open(experiment_name + '/results.txt', 'a')
        print('\n GENERATION ' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
            round(std, 6)))
        file_aux.write(
            '\n' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
        file_aux.close()

        wandb.log({'Fitness/Best': fit_pop[best], 
                   'Fitness/Mean': mean, 
                   'Fitness/Std': std, 
                   'PlayerHealth/Best': info_pop[best][0], 
                   'PlayerHealth/Mean': np.mean(info_pop[:, 0]),
                   'PlayerHealth/Std': np.std(info_pop[:, 0]),
                   'EnemyHealth/Best': info_pop[best][1], 
                   'EnemyHealth/Mean': np.mean(info_pop[:, 1]),
                   'EnemyHealth/Std': np.std(info_pop[:, 1]),
                   'Timesteps/Best': info_pop[best][2],
                   'Timesteps/Mean': np.mean(info_pop[:, 2]),
                   'Timesteps/Std': np.std(info_pop[:, 2]),
                   'Gain/Best': info_pop[best][0] - info_pop[best][1],
                   'Gain/Mean': np.mean(info_pop[:, 0] - info_pop[:, 1]),
                   'Gain/Std': np.std(info_pop[:, 0] - info_pop[:, 1])
                   }, step = i)

        # saves generation number
        file_aux = open(experiment_name + '/gen.txt', 'w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        file = gzip.open(experiment_name + '/best', 'wb', compresslevel=5)
        pickle.dump(best, file, protocol=2)  # type: ignore
        file.close()

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()

    fim = time.time()  # prints total execution time for experiment
    print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')
    print('\nExecution time: ' + str(round((fim - ini))) + ' seconds \n')

    best = int(np.argmax(fit_pop))  # best solution in generation
    for i in range(5):
        best_fit, best_info = evaluate([pop[best]])  # repeats best eval, for stability issues
        try:
            wandb.log({'Fitness5Times':best_fit[0], 
                    'PlayerHealth5Times': best_info[0][0], 
                    'EnemyHealth5Times': best_info[0][1], 
                    'Timesteps5Times': best_info[0][2],
                    'Gain5Times': best_info[0][0] - best_info[0][1],
                    }, step = i)
        except:
            print(best_fit, best_info)

    file = open(experiment_name + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

    env.state_to_log()  # checks environment state

def sigma_scaling(fit_pop, c=2):
    """Apply sigma scaling to the fitness values.

    Args:
        fit_pop (np.ndarray): fitness values of the population.
        c (int, optional): Scaling factor.

    Returns:
        np.ndarray: Scaled fitness values.
    """
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    scaled_fitness = 1 + (fit_pop - mean) / (c * std)

    # Prevent negative fitness values
    scaled_fitness = np.clip(scaled_fitness, 0.0000000001, None)

    return scaled_fitness


def print_dict(d: dict) -> None:
    """Print a dictionary

    Args:
        d (dict): Dictionary to print
    """
    for k, v in d.items():
        print(f'{k}: {v}')


def main(env: Environment, args: argparse.Namespace, cfg: dict) -> None:
    """Main function for the genetic algorithm

    Args:
        env (Environment): Environment object for the evoman framework
        args (argparse.Namespace): Command line arguments
        cfg (dict): Configuration dictionary
    """
    # loads file with the best solution for testing
    if args.run_mode == 'test':
        file = gzip.open(args.experiment_name + '/best')
        bsol = pickle.load(file, encoding='latin1')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed', 'normal')
        evaluate([bsol])
        sys.exit(0)

    # initializes population loading old solutions or generating new ones
    if not os.path.exists(args.experiment_name + '/evoman_solstate') or args.new_evolution:
        print('\nNEW EVOLUTION\n')
        pop, fit_pop, extra_info, fit_pop_stats, ini_g = generate_new_pop(cfg['npop'], cfg['archetecture'])

    else:
        print('\nCONTINUING EVOLUTION\n')
        pop, fit_pop, fit_pop_stats, ini_g = load_pop(env, args.experiment_name)

    # saves results for first pop
    best, mean, std = fit_pop_stats
    file_aux = open(experiment_name + '/results.txt', 'a')
    file_aux.write('\n\ngen best mean std')
    print('\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
        round(std, 6)))
    file_aux.write(
        '\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.close()

    train(env, pop, fit_pop, extra_info, best, ini_g, cfg)

def testbest(env: Environment, args: argparse.Namespace, cfg: dict) -> None:
    """Test the best found solution 5 times and logs the results to wandb

    Args:
        env (Environment): Environment object for the evoman framework
        args (argparse.Namespace): Command line arguments
        cfg (dict): Configuration dictionary
    """

    # loads file with the best solution for testing
    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = int(np.argmax(fit_pop))
    bsol = pop[best]

    print('\n RUNNING SAVED BEST SOLUTION \n')
    for i in range(5):
        fitness, other_info = evaluate([bsol])
        wandb.log({'Fitness5Times':fitness[0],
                     'PlayerHealth5Times': other_info[0][0],
                     'EnemyHealth5Times': other_info[0][1],
                     'Timesteps5Times': other_info[0][2],
                     'Gain5Times': other_info[0][0] - other_info[0][1],
                     })

if __name__ == "__main__":  # Basically just checks if the script is being run directly or imported as a module

    # Command line arguments, this makes it so you can run the script from the command line with different arguments as such:
    # python optimization_specialist_demo.py --experiment_name=optimization_test --headless=True --new_evolution=False --run_mode=train
    parser = argparse.ArgumentParser(description='Run the genetic algorithm for the evoman framework')
    parser.add_argument('--experiment_name', type=str, default='optimization_test', help='Name of the experiment')
    parser.add_argument('--headless', type=bool, default=True, help='Run the simulation without visuals')
    parser.add_argument('--new_evolution', type=bool, default=False, help='Start a new evolution')
    parser.add_argument('--run_mode', type=str, default='train', help='Run mode for the genetic algorithm')
    args = parser.parse_args()

    # Loading our training config
    cfg: dict = yaml.safe_load(open('./config/' + args.experiment_name + '.yaml'))  # type: ignore

    print('Config:')
    print_dict(cfg)

    # choose this for not using visuals and thus making experiments faster
    headless = args.headless
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"  # Turn off videodriver when running 
    
    for muttype in ['standard', 'non_uniform']:
        cfg['muttype'] = muttype
        enemies = [6,7,8]
        for enemy in enemies:
            for i in range(10):
                experiment_name = f'Evoman_Enemy{enemy}_NonUnMult{i}'

                # Create a folder to store the experiment
                if not os.path.exists(experiment_name):
                    os.makedirs(experiment_name)

                # Initialize Weights and Biases
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Evoman Project 1",

                    # track hyperparameters and run metadata
                    name=experiment_name,
                    config=cfg
                )

                

                # initializes simulation in individual evolution mode, for single static enemy.
                env = Environment(experiment_name=experiment_name,
                                enemies=[8],
                                playermode="ai",
                                player_controller=player_controller(cfg['archetecture']),
                                # Initialise player with specified archetecture
                                enemymode="static",
                                level=2,
                                speed="fastest",
                                visuals=False)

                env.state_to_log()  # checks environment state

                # Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
                ini = time.time()  # sets time marker

                main(env, args, cfg)
                wandb.finish()
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Evoman Project 1",

                    # track hyperparameters and run metadata
                    name=experiment_name+'_testBest',
                    config=cfg
                )
                testbest(env, args, cfg)
                wandb.finish()
                
