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
    return x.clip(min=0.000000001)
    

# runs simulation
def simulation(env,x):
    f,p,e,t,g,de = env.play(pcont=x)
    return f, p, e, t, g, de

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
    results = np.array([simulation(env, individual) for individual in x])
    fitness = results[:, 0]
    other_info = results[:, 1:]
    return fitness, other_info


# tournament
def tournament(pop: list[list[tuple[np.ndarray, np.ndarray]]], fit_pop: np.ndarray, size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tournament function to select the fittest of two individuals in the population

    Args:
        pop (list[list[tuple[np.ndarray, np.ndarray]]]): Population of shape (pop_size*layer_amt*2)
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)

    Returns:
        list(tuple(weigths, bias)): One individual in the population
    """

    # Get two random values that correspond to two individuals in the population
    c = np.random.randint(0, len(pop), size=size)

    # Get the index of the fittest individual
    maxIndex = c[np.argmax(fit_pop[c])]

    # tdiff = 0
    # for l1, l2 in zip(pop[c1], pop[c2]):
    #     wdiff = np.sum(np.abs(l1[1] - l2[1]))
    #     bdiff = np.sum(np.abs(l1[0] - l2[0]))
    #     tdiff += (wdiff + bdiff)*0.01

    return pop[maxIndex]


def mutate(vals: np.ndarray, probability: float) -> np.ndarray:
    """Mutate the values of a layer

    Args:
        vals (np.ndarray): Weights or biases of a layer

    Returns:
        np.ndarray: Mutated weights or biases
    """
    # TODO: Add non-linearity to the mutation
    # TODO: Add swap mutation
    mutation_mask = np.random.uniform(0, 1, size=vals.shape) <= probability
    vals[mutation_mask] += np.random.normal(0, 1, size=vals[mutation_mask].shape)

    return vals



# crossover
def crossover(pop: list[list[tuple[np.ndarray, np.ndarray]]], fit_pop: np.ndarray, method: int, cfg: dict) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Crossover function to generate offspring from the population

    Args:
        pop (list[tuple[np.ndarray, np.ndarray]]): Population formatted as a an array of models containing tuples of weights and biases.
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)

    Returns:
        np.ndarray: New population of offspring
    """    
    # TODO: Add incest, blijkbaar toch niet

    # Goes through pairs in the population and chooses two random fit individuals according to tournament
    probs = softmax_activation(fit_pop)
    for p in range(0,len(pop), 2):
        chosen = np.random.choice(len(pop), 2 , p=probs, replace=False)
        p1, p2 = pop[chosen[0]], pop[chosen[1]]

        n_offspring =   np.random.randint(1, 6)
        offspring: list[list[tuple[np.ndarray, np.ndarray]]] = []

        for f in range(n_offspring):
            cross_prop = np.random.uniform(0,1) # Get a random ratio of influence between parents p1 and p2
            child: list[tuple[np.ndarray, np.ndarray]] = []

            for layer_p1, layer_p2 in zip(p1, p2):
                # crossover
                weightsp = np.random.uniform(0,1)
                biasp = np.random.uniform(0,1)
                if method == 0:
                    weights, bias = snipcombine(layer_p1, layer_p2)
                else:
                    weights, bias = blendcombine(layer_p1, layer_p2, cross_prop)

                # weights = layer_p1[1]*cross_prop+layer_p2[1]*(1-cross_prop)
                # bias = layer_p1[0]*cross_prop+layer_p2[0]*(1-cross_prop)
                # weights = (layer_p1[1], layer_p2[1])[weightsp > cross_prop]
                # bias = (layer_p1[0], layer_p2[0])[biasp > cross_prop]

                # mutation
                weights = mutate(weights, cfg['mutation'])
                bias = mutate(bias, cfg['mutation'])

                # limit between -1 and 1 using Tanh function
                weights = tanh_activation(weights)
                bias = tanh_activation(bias)
                
                child.append((weights , bias))

            offspring.append(child)

    return offspring

def snipcombine(layer_p1, layer_p2):
    weightspoint = np.random.uniform(0,1) * len(layer_p1[0])
    biaspoint = np.random.uniform(0,1) * len(layer_p1[1])

                # Snip parts of parents together
    weights = np.concatenate((layer_p1[0][:int(weightspoint)], layer_p2[0][int(weightspoint):]))
    bias = np.concatenate((layer_p1[1][:int(biaspoint)], layer_p2[1][int(biaspoint):]))
    return weights,bias

def blendcombine(layer_p1, layer_p2, cross_prob):
    weights = layer_p1[0]*cross_prob+layer_p2[0]*(1-cross_prob)
    bias = layer_p1[1]*cross_prob+layer_p2[1]*(1-cross_prob)
    return weights, bias

def doomsday(pop: list[list[tuple[np.ndarray, np.ndarray]]],fit_pop:np.ndarray, other_info: np.ndarray, cfg: dict) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], np.ndarray, np.ndarray]:
    """Kills the worst genomes, and replace with new best/random solutions

    Args:
        pop (list[tuple[np.ndarray, np.ndarray]]): Population formatted as a an array of models containing tuples of weights and biases.
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: New population
    """

    worst_indices = np.argsort(fit_pop)[(cfg['npop'] // 4)*3:] # A quarter of the individuals
    best_individual = pop[np.argmax(fit_pop)]

    assert np.argmax(fit_pop) not in worst_indices, "Best individual is also one of the worst"

    # Nuke the hell out of the worst, make them mutate like crazy
    for idx in worst_indices:
        for layer_idx, (weights, biases) in enumerate(pop[idx]):
            nuke(cfg, weights, biases)

            pop[idx][layer_idx] = (weights, biases)

        fit_pop[idx], other_info[idx] = evaluate([pop[idx]])

    return pop, fit_pop, other_info

def nuke(cfg, weights, biases):
    mutation_mask = np.random.uniform(0, 1, size=weights.shape) <= np.random.uniform(0, 1)
    mutation_mask_b = np.random.uniform(0, 1, size=biases.shape) <= np.random.uniform(0, 1)
    weights[mutation_mask] = np.random.uniform(cfg['dom_l'], cfg['dom_u'], size=weights[mutation_mask].shape)
    biases[mutation_mask_b] = np.random.uniform(cfg['dom_l'], cfg['dom_u'], size=biases[mutation_mask_b].shape)

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

def make_species(pop: list[list[tuple[np.ndarray, np.ndarray]]], fit_pop: np.ndarray, other_info: np.ndarray, cfg: dict) -> tuple[list[list[list[tuple[np.ndarray, np.ndarray]]]], list[np.ndarray], list[np.ndarray]]:
    """Split the population into species

    Args:
        pop (list[list[tuple[np.ndarray, np.ndarray]]]): Population formatted as a an array of models containing tuples of weights and biases.
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)
        other_info (np.ndarray): Other information about the individuals in the population in shape (player_energy, enemy_energy, timesteps)
        cfg (dict): Configuration dictionary

    Returns:
        tuple[list[list[list[tuple[np.ndarray, np.ndarray]]]], list[np.ndarray], list[np.ndarray]]: Species population, fitness values, and other information
    """
    species_pop: list[list[list[tuple[np.ndarray, np.ndarray]]]] = []
    species_fit: list[np.ndarray] = []
    species_other: list[np.ndarray] = []
    spec_size = cfg['npop'] // cfg['species']
    for i in range(cfg['species']):
        species_pop.append(pop[i*spec_size:(i+1)*spec_size])
        species_fit.append(fit_pop[i*spec_size:(i+1)*spec_size])
        species_other.append(other_info[i*spec_size:(i+1)*spec_size])
    
    return species_pop, species_fit, species_other

def cross_species(species_pop: list[list[list[tuple[np.ndarray, np.ndarray]]]], species_fit: list[np.ndarray], species_other: list[np.ndarray], cfg: dict) -> tuple[list[list[list[tuple[np.ndarray, np.ndarray]]]], list[np.ndarray], list[np.ndarray]]:
    """Crossover between species

    Args:
        species_pop (list[list[list[tuple[np.ndarray, np.ndarray]]]]): Species population
        species_fit (list[np.ndarray]): Fitness values for each individual in the species population
        cfg (dict): Configuration dictionary

    Returns:
        list[list[list[tuple[np.ndarray, np.ndarray]]]]: New species population
    """
    prob = random.uniform(0,1)
    if cfg['spec_cross'] >= prob:
        print("Crossing species")
        for target in range(cfg['species']-1):
            origin_index = (target + 1) % cfg['species']
            origin_fit = species_fit[origin_index]

            target_fit = species_fit[target]

            best_origin_ind = np.argmax(origin_fit)
            best_target_ind = np.argmax(target_fit)

            # Swap best of the two species
            species_pop[target][best_target_ind], species_pop[origin_index][best_origin_ind] = species_pop[origin_index][best_origin_ind], species_pop[target][best_target_ind]
            species_fit[target][best_target_ind], species_fit[origin_index][best_origin_ind] = species_fit[origin_index][best_origin_ind], species_fit[target][best_target_ind]
            species_other[target][best_target_ind], species_other[origin_index][best_origin_ind] = species_other[origin_index][best_origin_ind], species_other[target][best_target_ind]
            
        return species_pop, species_fit, species_other
    
    return species_pop, species_fit, species_other

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
    c = 2.
    notimproved = 0

    species_pop, species_fit, species_other = make_species(pop, fit_pop, other_pop, cfg)
    best_sols = [0]*cfg['species']
    last_sols = [0]*cfg['species']
    spec_notimproved = [0]*cfg['species']

    for g in range(ini_g+1, cfg['gens']):
        for i, popinfo in enumerate(zip(species_pop, species_fit, species_other)):
            pop, fit, other = popinfo
            offspring = crossover(pop, fit, cfg['comb_meths'][i], cfg)  # crossover
            fit_offspring, other_info = evaluate(offspring)   # evaluation
            pop = pop + offspring
            fit = np.append(fit,fit_offspring)
            other = np.append(other_pop, other_info, axis=0)
            # Add sigma scaling to fitness values
            
            
            best = int(np.argmax(fit)) #best solution in generation
            fit[best] = float(evaluate([pop[best] ])[0][0]) # repeats best eval, for stability issues
            best_sols[i] = fit[best]

            fit_pop = sigscaler(fit, c)

            # selection
            fit_pop_cp = fit_pop.copy()
            fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
            probs = (fit_pop_norm)/(fit_pop_norm).sum() # normalize fitness values to probabilities
            # Assert there is no NaN in fit_pop
            assert not np.isnan(fit_pop).any()
            # Assert there are no negatives
            assert not (fit_pop < 0).any()
            # probs = softmax_activation(fit_pop)
            chosen = np.random.choice(len(pop), cfg['npop'] // cfg['species'] , p=probs, replace=False)
            chosen = np.append(chosen[1:],best)

            newpop: list[list[tuple[np.ndarray, np.ndarray]]] = []
            for c in chosen:
                newpop.append(pop[c])
            pop = newpop

            newfit = []
            newraw = []
            newinfo = []
            for c in chosen:
                newinfo.append(other[c])
                newraw.append(fit[c])
                newfit.append(fit_pop[c])
            fit_pop = np.array(newfit)
            fit = np.array(newraw)
            other = np.array(newinfo)


            # searching new areas

            if best_sols[i] <= last_sols[i]:
                spec_notimproved[i] += 1
            else:
                last_sols[i] = best_sols[i]
                spec_notimproved[i] = 0

            if spec_notimproved[i] >= cfg['doomsteps']:

                file_aux  = open(experiment_name+'/results.txt','a')
                file_aux.write('\ndoomsday')
                file_aux.close()
                assert other.shape[0] == fit.shape[0]
                pop, fit, other = doomsday(pop,fit, other, cfg)
                # fit_raw, other_pop = evaluate(pop)
                # fit_pop = sigscaler(fit.copy(), c)
                spec_notimproved[i] = 0
            
            species_fit[i] = fit
            species_other[i] = other
            species_pop[i] = pop

        fit_raw = [f for fit in species_fit for f in fit]
        pop = [ind for species in species_pop for ind in species]
        other_pop = np.array([info for species in species_other for info in species])

        best = int(np.argmax(fit_raw))
        std  =  np.std(fit_raw)
        mean = np.mean(fit_raw)
        # c += 0.05


        # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(g)+' '+str(round(fit_raw[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        wandb.log({'Generation': ini_g, 'Best Fitness': fit_raw[best], 'Mean Fitness': mean, 'Std Fitness': std, 'Best Player Health': other_pop[best][0], 'Best Enemy Health': other_pop[best][1], 'Best Timesteps': other_pop[best][2],'Gain': other_pop[best][3], 'Best Dead Enemies': other_pop[best][4]})
        file_aux.write('\n'+str(g)+' '+str(round(fit_raw[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
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

        species_pop, species_fit, species_other = cross_species(species_pop, species_fit, species_other, cfg)

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
                    enemies=[1,2,3,4,5,6,7,8],
                    multiplemode="yes",
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