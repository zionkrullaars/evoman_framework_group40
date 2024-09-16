###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from wouters_test_controller import player_controller

# imports other libs
import time
import numpy as np
import random
from math import fabs,sqrt
import glob, os
import gzip, pickle, yaml
import argparse
import wandb
from nnlayers import tanh_activation, sigmoid_activation, softmax_activation

np.random.seed(42)
random.seed(42)

def sigscaler(x: np.ndarray, c: float) -> np.ndarray:
    x = x - (x.mean() - c * x.std())
    return x.clip(min=0)
    

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f, p, e, t

# normalizes
def norm(x, pfit_pop):

    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


# evaluation
def evaluate(x: list[list[tuple[np.ndarray, np.ndarray]]]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the fitness of a population of individuals

    Args:
        x (list[list[tuple[np.ndarray, np.ndarray]]]): Population of individuals, each individual is a list of tuples containing weights and biases for each layer

    Returns:
        np.ndarray: List of fitness values for each individual in the population of size (pop_size*1)
        np.ndarray: Other information about the individuals in the population in shape (player_energy, enemy_energy, timesteps)
    """
    runs = list(map(lambda y: simulation(env,y), x))
    fitness = np.array([r[0] for r in runs])
    other_info = np.array([r[1:] for r in runs])
    return fitness, other_info


# tournament
def tournament(pop: list[list[tuple[np.ndarray, np.ndarray]]], fit_pop: np.ndarray, repeat: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tournament function to select the fittest of two individuals in the population

    Args:
        pop (list[list[tuple[np.ndarray, np.ndarray]]]): Population of shape (pop_size*layer_amt*2)
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)

    Returns:
        list(tuple(weigths, bias)): One individual in the population
    """

    # Get two random values that correspond to two individuals in the population
    c1 =  random.randint(0,len(pop)-1)

    # Repeat the tournament for a set amount of times, c1 will be the fittest of the two
    for i in range(repeat):
        c2 =  random.randint(0,len(pop)-1)

        tdiff = 0
        for l1, l2 in zip(pop[c1], pop[c2]):
            wdiff = np.sum(np.abs(l1[1] - l2[1]))
            bdiff = np.sum(np.abs(l1[0] - l2[0]))
            tdiff += (wdiff + bdiff)*0.01


        # Return the fittest of the two, also prioritise genetic diversity
        if fit_pop[c1] > fit_pop[c2]:
            c1 = c2
        else:
            c1 = c1

    return pop[c1]


def mutate(vals: np.ndarray, probability: float) -> np.ndarray:
    """Mutate the values of a layer

    Args:
        vals (np.ndarray): Weights or biases of a layer

    Returns:
        np.ndarray: Mutated weights or biases
    """
    # TODO: Add non-linearity to the mutation
    # TODO: Add swap mutation
    for i in range(0,len(vals)):
        if np.random.uniform(0 ,1)<=probability:
            vals[i] =   vals[i]+np.random.normal(0, 1)

    return vals

# crossover
def crossover(pop: list[list[tuple[np.ndarray, np.ndarray]]], fit_pop: np.ndarray, cfg: dict) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Crossover function to generate offspring from the population

    Args:
        pop (list[tuple[np.ndarray, np.ndarray]]): Population formatted as a an array of models containing tuples of weights and biases.
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)

    Returns:
        np.ndarray: New population of offspring
    """    
    # TODO: Add incest

    # Goes through pairs in the population and chooses two random fit individuals according to tournament
    for p in range(0,len(pop), 2):
        p1 = tournament(pop, fit_pop,1)
        p2 = tournament(pop, fit_pop,1)

        n_offspring =   np.random.randint(1,3+1, 1)[0]
        offspring: list[list[tuple[np.ndarray, np.ndarray]]] = []

        for f in range(0,n_offspring):
            cross_prop = np.random.uniform(0,1) # Get a random ratio of influence between parents p1 and p2
            child: list[tuple[np.ndarray, np.ndarray]] = []

            for layer_p1, layer_p2 in zip(p1, p2):
                # crossover
                weightsp = np.random.uniform(0,1)
                biasp = np.random.uniform(0,1)
                weights = layer_p1[1]*cross_prop+layer_p2[1]*(1-cross_prop)
                bias = layer_p1[0]*cross_prop+layer_p2[0]*(1-cross_prop)
                # weights = (layer_p1[1], layer_p2[1])[weightsp > cross_prop]
                # bias = (layer_p1[0], layer_p2[0])[biasp > cross_prop]

                # mutation
                weights = mutate(weights, cfg['mutation'])
                bias = mutate(bias, cfg['mutation'])

                # limit between -1 and 1 using Tanh function
                bias = tanh_activation(bias)
                weights = tanh_activation(weights)
                
                child.append((bias, weights))

            offspring.append(child)

    return offspring


def doomsday(pop: list[list[tuple[np.ndarray, np.ndarray]]],fit_pop:np.ndarray, cfg: dict) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Kills the worst genomes, and replace with new best/random solutions

    Args:
        pop (list[tuple[np.ndarray, np.ndarray]]): Population formatted as a an array of models containing tuples of weights and biases.
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: New population
    """

    worst = int(cfg['npop']//4)  # a quarter of the population
    order = np.argsort(fit_pop) # sort the population by fitness
    orderasc = order[0:worst] # get the worst individuals

    for o in orderasc:
        for l, layer in enumerate(pop[o]):
            for v, vect in enumerate(layer): # Go through the bias and weights vectors
                for i in range(0,len(vect)):
                    pro = np.random.uniform(0,1) # Get a random probability
                    if np.random.uniform(0,1)  <= pro:
                        pop[o][l][v][i] = np.random.uniform(cfg['dom_l'], cfg['dom_u']) # random dna, uniform dist.
                    else:
                        pop[o][l][v][i] = pop[order[-1:][0]][l][v][i] # dna from best, which is the last index (-1) of the order list

        
        # val = evaluate([pop[o]]) # Evaluate the new individual
        # fit_pop[o]=val[0]

    return pop

def generate_new_pop(npop: int, n_hidden_neurons: list[int]) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], np.ndarray, np.ndarray, tuple[int, float, float], int]:
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
        in_size = 20 # Amount of input neurons (sensors of the game)
        for layer_size in n_hidden_neurons:
            weights = np.random.uniform(-1,1,(in_size, layer_size))
            bias = np.random.uniform(-1,1,(1, layer_size))
            in_size = layer_size
            individual.append((weights, bias))
        pop.append(individual)

    ###############################################################################
    # Zo ziet pop er uit!
    # [
    #     [                                      Model 1
    #         ([1,1,...,0.5], [1,2,...,0.1])     Linear layer 1 (weights, bias)
    #     ],
    # ]
    ###############################################################################
        
    fit_pop, other_info = evaluate(pop)
    best = int(np.argmax(fit_pop))
    mean = float(np.mean(fit_pop))
    std = float(np.std(fit_pop))
    ini_g = int(0)
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    return pop, fit_pop, other_info, (best, mean, std), ini_g

def load_pop(env: Environment, experiment_name: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, tuple[int, float, float], int]:
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
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()
    return pop, fit_pop, (best, mean, std), ini_g

def train(env: Environment, pop: list[list[tuple[np.ndarray, np.ndarray]]], fit_pop: np.ndarray, other_pop: np.ndarray, best: int, ini_g: int, cfg: dict) -> None:
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
    fit_raw = fit_pop.copy()
    c = 0.2
    notimproved = 0

    for i in range(ini_g+1, cfg['gens']):

        offspring = crossover(pop, fit_pop, cfg)  # crossover
        fit_offspring, other_info = evaluate(offspring)   # evaluation
        pop = pop + offspring
        fit_raw = np.append(fit_raw,fit_offspring)
        fit_pop = np.append(fit_pop,fit_offspring)
        other_pop = np.append(other_pop, other_info, axis=0)
        # Add sigma scaling to fitness values
        fit_pop = sigscaler(fit_pop, c)
        
        best = int(np.argmax(fit_pop)) #best solution in generation
        fit_pop[best] = float(evaluate([pop[best] ])[0][0]) # repeats best eval, for stability issues
        best_sol = fit_pop[best]

        # selection
        fit_pop_cp = fit_pop
        fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
        probs = (fit_pop_norm)/(fit_pop_norm).sum() # normalize fitness values to probabilities
        # probs = softmax_activation(fit_pop)
        chosen = np.random.choice(len(pop), cfg['npop'] , p=probs, replace=False)
        chosen = np.append(chosen[1:],best)

        newpop: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for c in chosen:
            newpop.append(pop[c])
        pop = newpop

        newfit = []
        for c in chosen:
            newfit.append(fit_pop[c])
        fit_pop = np.array(newfit)


        # searching new areas

        if best_sol <= last_sol:
            notimproved += 1
        else:
            last_sol = best_sol
            notimproved = 0

        if notimproved >= 15:

            file_aux  = open(experiment_name+'/results.txt','a')
            file_aux.write('\ndoomsday')
            file_aux.close()

            pop = doomsday(pop,fit_pop, cfg)
            fit_raw, other_pop = evaluate(pop)
            fit_pop = sigscaler(fit_raw.copy(), c)
            notimproved = 0

        best = int(np.argmax(fit_raw))
        std  =  np.std(fit_raw)
        mean = np.mean(fit_raw)
        c += 0.05


        # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(i)+' '+str(round(fit_raw[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        wandb.log({'Generation': ini_g, 'Best Fitness': fit_raw[best], 'Mean Fitness': mean, 'Std Fitness': std, 'Best Player Health': other_pop[best][0], 'Best Enemy Health': other_pop[best][1], 'Best Timesteps': other_pop[best][2]})
        file_aux.write('\n'+str(i)+' '+str(round(fit_raw[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()

        # saves generation number
        file_aux  = open(experiment_name+'/gen.txt','w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        file = gzip.open(experiment_name+'/best', 'wb', compresslevel = 5)
        pickle.dump(best, file, protocol=2) # type: ignore
        file.close()

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()

    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')


    file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

    env.state_to_log() # checks environment state

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
    if args.run_mode =='test':
        file = gzip.open(args.experiment_name+'/best')
        bsol =  pickle.load(file, encoding='latin1')
        print( '\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed','normal')
        evaluate([bsol])
        sys.exit(0)


    # initializes population loading old solutions or generating new ones
    if not os.path.exists(args.experiment_name+'/evoman_solstate') or args.new_evolution:
        print( '\nNEW EVOLUTION\n')
        pop, fit_pop, other_info, fit_pop_stats, ini_g = generate_new_pop(cfg['npop'], cfg['archetecture'])

    else:
        print( '\nCONTINUING EVOLUTION\n')
        pop, fit_pop, fit_pop_stats, ini_g = load_pop(env, args.experiment_name)

    # saves results for first pop
    best, mean, std = fit_pop_stats
    file_aux  = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen best mean std')
    print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    train(env, pop, fit_pop, other_info, best, ini_g, cfg)

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
    env = Environment(experiment_name=experiment_name,
                    enemies=[8],
                    playermode="ai",
                    player_controller=player_controller(cfg['archetecture']), # Initialise player with specified archetecture
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

    env.state_to_log() # checks environment state


    ####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
    ini = time.time()  # sets time marker

    main(env, args, cfg)