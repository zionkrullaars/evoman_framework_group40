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


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = [20,10,5]



# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
# n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
n_vars = env.get_num_sensors() + 1
for i in range(len(n_hidden_neurons)):
    n_vars = n_vars*n_hidden_neurons[i]

for i in range(len(n_hidden_neurons)):
    n_vars += n_hidden_neurons[i]


dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation = 0.2
last_best = 0


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

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
def evaluate(x: list[list[tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
    """Evaluate the fitness of a population of individuals

    Args:
        x (list[list[tuple[np.ndarray, np.ndarray]]]): Population of individuals, each individual is a list of tuples containing weights and biases for each layer

    Returns:
        np.ndarray: List of fitness values for each individual in the population of size (pop_size*1)
    """
    return np.array(list(map(lambda y: simulation(env,y), x)))


# tournament
def tournament(pop: list[list[tuple[np.ndarray, np.ndarray]]], fit_pop: np.ndarray) -> np.ndarray:
    """Tournament function to select the fittest of two individuals in the population

    Args:
        pop (list[list[tuple[np.ndarray, np.ndarray]]]): Population of shape (pop_size*layer_amt*2)
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)

    Returns:
        list(tuple(weigths, bias)): One individual in the population
    """

    # Get two random values that correspond to two individuals in the population
    c1 =  random.randint(0,len(pop)-1)
    c2 =  random.randint(0,len(pop)-1)

    # Return the fittest of the two 
    if fit_pop[c1] > fit_pop[c2]:
        return pop[c1]
    else:
        return pop[c2]


# limits
def limits(x):

    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x

# [
#     [                             # Model 1
#         ([1,1,0.5], [1,2,0.1])    # Linear layer 1
#     ]
# ]

# crossover
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
        if np.random.uniform(0 ,1)<=mutation:
            vals[i] =   vals[i]+np.random.normal(0, 1)

    return vals


def crossover(pop: list[list[tuple[np.ndarray, np.ndarray]]], fit_pop: np.ndarray) -> np.ndarray:
    """Crossover function to generate offspring from the population

    Args:
        pop (list[tuple[np.ndarray, np.ndarray]]): Population formatted as a an array of models containing tuples of weights and biases.
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)

    Returns:
        _type_: _description_
    """    
    # TODO: Add incest
    total_offspring = np.zeros((0,n_vars))

    # Goes through pairs in the population and chooses two random fit individuals according to tournament
    for p in range(0,len(pop), 2):
        print(len(pop[0]))
        p1 = tournament(pop, fit_pop)
        p2 = tournament(pop, fit_pop)

        n_offspring =   np.random.randint(1,3+1, 1)[0]
        offspring: list[tuple[np.ndarray, np.ndarray]] = []


        for f in range(0,n_offspring):
            cross_prop = np.random.uniform(0,1) # Get a random ratio of influence between parents p1 and p2
            child = []

            for layer_p1, layer_p2 in zip(p1, p2):
                # crossover
                
                weights = layer_p1[1]*cross_prop+layer_p2[1]*(1-cross_prop)
                bias = layer_p1[0]*cross_prop+layer_p2[0]*(1-cross_prop)

                # mutation
                weights = mutate(weights, mutation)
                bias = mutate(bias, mutation)

                # limit between -1 and 1
                # TODO: Do this through non-linear tanh function (see nnlayers.py for other non linear functions)
                bias = bias.clip(-1, 1)
                weights = weights.clip(-1, 1)
                
                child.append((bias, weights))

            offspring.append(child)

    return offspring


def doomsday(pop: list[tuple[np.ndarray, np.ndarray]],fit_pop:np.ndarray, npop: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Kills the worst genomes, and replace with new best/random solutions

    Args:
        pop (list[tuple[np.ndarray, np.ndarray]]): Population formatted as a an array of models containing tuples of weights and biases.
        fit_pop (np.ndarray): Array of fitness values for each individual in population, of shape (pop_size*1)

    Returns:
        _type_: _description_
    """

    worst = int(npop//4)  # a quarter of the population
    order = np.argsort(fit_pop) # sort the population by fitness
    orderasc = order[0:worst] # get the worst individuals

    for o in orderasc:
        for l, layer in enumerate(pop[o]):
            for v, vect in enumerate(layer): # Go through the bias and weights vectors
                for i in range(0,len(vect)):
                    pro = np.random.uniform(0,1) # Get a random probability
                    if np.random.uniform(0,1)  <= pro:
                        pop[o][l][v][i] = np.random.uniform(dom_l, dom_u) # random dna, uniform dist.
                    else:
                        pop[o][l][v][i] = pop[order[-1:]][l][v][i] # dna from best, which is the last index (-1) of the order list

        fit_pop[o]=evaluate([pop[o]]) # Evaluate the new individual

    return pop,fit_pop



# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    # pop = np.random.uniform(len((n_hidden_neurons), dom_l, dom_u, (npop, n_vars))
    pop: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for i in range(npop):
        individual = []
        in_size = 20
        for layer_size in n_hidden_neurons:
            weights = np.random.uniform(in_size, layer_size)
            bias = np.random.uniform(1, layer_size)
            in_size = layer_size
            individual.append((bias, weights))
        pop.append(individual)
        
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()




# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()


# evolution

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g+1, gens):

    offspring = crossover(pop, fit_pop)  # crossover
    fit_offspring = evaluate(offspring)   # evaluation
    pop = pop + offspring
    fit_pop = np.append(fit_pop,fit_offspring)

    best = np.argmax(fit_pop) #best solution in generation
    fit_pop[best] = float(evaluate([pop[best] ])[0]) # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    # selection
    # TODO: Add sigma scaling
    fit_pop_cp = fit_pop
    fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
    probs = (fit_pop_norm)/(fit_pop_norm).sum() # normalize fitness values to probabilities
    chosen = np.random.choice(len(pop), npop , p=probs, replace=False)
    chosen = np.append(chosen[1:],best)
    pop = pop[chosen]
    fit_pop = fit_pop[chosen]


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

        pop, fit_pop = doomsday(pop,fit_pop)
        notimproved = 0

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])

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
