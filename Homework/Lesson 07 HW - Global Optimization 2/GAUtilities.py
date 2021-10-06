import importlib
import numpy as np

def computeFitness(f, pop, **kwargs):
    '''
    Computes fitness based on passed in function.
    
    Parameters:
    f (function, required): the function used to evaluate fitness
    pop (numpy array, required): the population on which to evaluate fitness - individuals in rows.
    **kwargs (named arguments, optional): additional arguments that will be passed through to the fitness function
    
    Returns a numpy array of computed fitnesses.
    '''  
    #get sizes from pop
    pop_size, ind_size = pop.shape[0], pop.shape[1]
    
    #create the fitness array of zeros
    fitness = np.zeros(pop_size)
    #fill the fitness array
    for j in range(pop_size):
        fitness[j] = f(pop[j], **kwargs)
        
    return fitness    

# Sort population
def sortPop(pop, fitness):
    '''
    Sorts a population with minimal fitness in first place.
    
    Parameters:
    pop (numpy array, required): The population to sort, individuals in rows
    fitness (numpy array, required): The values used to sort the population
    
    Returns:
    Sorted numpy array of population
    '''
    #sort by increasing fitness
    sorted_pos = fitness.argsort() 
    fitness = fitness[sorted_pos]
    pop = pop[sorted_pos]
    return pop.copy()

####################################
# Selection operators
####################################

# tournament selection
def tournamentSelection(pop, tourn_size, debug=False):
    '''
    Implements tournameent selection on a population.
    
    Parameters:
    pop (numpy array, required): The sorted population from which selections will be drawn.
    tourn_size (between 2 and population size, required): The number of individuals that will compete in each tournament
    debug (boolean, optional, default=False): Flag to indicate whether to output debugging print statements.
    
    Returns:
    Numpy Array of the selected population.
    
    '''
    #get the population size
    pop_size, ind_size = pop.shape[0], pop.shape[1]

    # initialize selected population
    select_pop = np.zeros((pop_size,ind_size)) 
    for j in range(pop_size):
        subset_pos = np.random.choice(pop_size,tourn_size,replace=False) # select without replacement
        smallest_pos = np.min(subset_pos) # choose index corresponding to lowest fitness
        if debug:
            print('Individuals in tournament:', subset_pos)
            print('Individual selected:', smallest_pos)
        select_pop[j] = pop[smallest_pos]
    return select_pop     

####################################
# Crossover operators
####################################

def orderedCrossover(pop, cx_prob, debug=False):
    '''
    Peforms ordered crossover on permutation populations.
    
    Parameters:
    pop (numpy array of permutations, required): The population of permutations, individuals as rows
    cx_prob (real between 0 and 1, required): The probability that any two individuals will mate
    debug (boolean, optional, default=False): Flag to indicate whether to output debugging print statements.
    
    Returns: 
    Population of integers
    '''
    #get the sizes from population
    pop_size, ind_size = pop.shape[0], pop.shape[1]
    cx_pop = np.zeros((pop_size,ind_size)) # initialize crossover population
    for j in range(int(pop_size/2)):  # pop_size must be even
        parent1, parent2 = pop[2*j], pop[2*j+1]
        child1, child2 = parent2.copy(), parent1.copy()
        if np.random.uniform() < cx_prob: # crossover occurs
            swap_idx = np.sort(np.random.randint(0,ind_size,2))
            hole = np.full( ind_size, False, dtype = bool)
            hole[swap_idx[0]:swap_idx[1]+1] = True
            child1[~hole] = np.array([x for x in parent1 if x not in parent2[hole]])
            child2[~hole] = np.array([x for x in parent2 if x not in parent1[hole]])
            if debug:
                print("Crossover happened for individual", j)
                print('Parent 1', parent1)
                print('Parent 2', parent2)
                print('Swap Index:', swap_idx)
                print('Hole', hole)
                print('Child1', child1)
                print('Child2', child2)             
        cx_pop[2*j] = child1
        cx_pop[2*j+1] = child2
    return cx_pop.astype(int)


def onePointCrossover(pop, cx_prob, debug=False):
    '''
    Peforms one-point crossover on integer, boolean or real populations.
    
    Parameters:
    pop (numpy array, required): The population, individuals as rows
    cx_prob (real between 0 and 1, required): The probability that any two individuals will mate
    debug (boolean, optional, default=False): Flag to indicate whether to output debugging print statements.
    
    Returns: 
    Population (will return as reals, should be cast if integer or boolean is desired)
    '''
    # one-point crossover (mating)
    #get the sizes from pop
    pop_size, ind_size = pop.shape[0], pop.shape[1]
    cx_pop = np.zeros((pop_size,ind_size)) # initialize crossover population
    for j in range(int(pop_size/2)):  # pop_size must be even
        parent1, parent2 = pop[2*j], pop[2*j+1]
        child1, child2 = parent1.copy(), parent2.copy()
        if np.random.uniform() < cx_prob: # crossover occurs
            cx_point = np.random.randint(1,ind_size-1) # crossover point between 1 and ind_size-1 (if cx happens at end points, nothing happens)
            if debug:
                print('Crossover happened between Individuals', 2*j, 'and', 2*j+1, 'at point', cx_point)
            child1[0:cx_point], child2[0:cx_point] = parent2[0:cx_point], parent1[0:cx_point]
        cx_pop[2*j] = child1
        cx_pop[2*j+1] = child2
    return cx_pop

def blendedCrossover(pop, cx_prob, alpha, bounds, debug=False):
    '''
    Peforms blended crossover on real populations.
    
    Parameters:
    pop (numpy array, required): The population, individuals as rows
    cx_prob (real between 0 and 1, required): The probability that any two individuals will mate
    alpha (real, required): the amount of expansion
    bounds (list, required): the [lower, upper] bounds of possible values for each gene
    debug (boolean, optional, default=False): Flag to indicate whether to output debugging print statements.
    
    Returns: 
    Population of real variables
    '''
    #get individual size and population size
    pop_size, ind_size = pop.shape[0], pop.shape[1]
    cx_pop = np.zeros((pop_size,ind_size)) # initialize crossover population
    for j in range(int(pop_size/2)):  # pop_size must be even
        parent1, parent2 = pop[2*j], pop[2*j+1]
        child1, child2 = parent1.copy(), parent2.copy()
        if np.random.uniform() < cx_prob: # crossover occurs
            if debug:
                print('Crossover occurred between', 2*j, 'and', 2*j+1)
            for i, (x1, x2) in enumerate(zip(child1, child2)):
                l = min(x1,x2)
                r = max(x1,x2)
                bb = np.clip(np.random.uniform(l-alpha*(x2-x1),r+alpha*(x2-x1),size=2),bounds[0],bounds[1])
                child1[i] = bb[0]
                child2[i] = bb[1]
        cx_pop[2*j] = child1
        cx_pop[2*j+1] = child2
    return cx_pop 



####################################
# Mutation operators
####################################
def gaussianMutation(pop, mut_prob, ind_prob, bounds, sigma, debug=False): 
    '''
    Peforms gaussian mutation on real populations.
    
    Parameters:
    pop (numpy array, required): The population, individuals as rows
    mut_prob (real between 0 and 1, required): The probability that any individual will mutate
    ind_prob (real between 0 and 1, required): The probability that a gene will mutate
    bounds (list, required): the [lower, upper] bounds of possible values for each gene
    sigma (real, required): standard deviation used to generate new mutations
    debug (boolean, optional, default=False): Flag to indicate whether to output debugging print statements.
    
    Returns: 
    Population of real variables
    '''
    #get individual size and population size
    pop_size, ind_size = pop.shape[0], pop.shape[1]
    mut_pop = np.zeros((pop_size,ind_size)) # initialize mutation population
    for j in range(pop_size):
        individual = pop[j].copy() # copy is necessary to avoid conflicts in memory
        if np.random.uniform()<mut_prob:
            if debug:
                print("Mutation probability met for individual", j)
            individual = individual + np.random.normal(0,sigma,ind_size)*(np.random.uniform(size=ind_size)<ind_prob)
            individual = np.maximum(individual,bounds[0]) # clip to lower bound
            individual = np.minimum(individual,bounds[1]) # clip to upper bound
        mut_pop[j] = individual.copy() # copy is necessary to avoid conflicts in memory
    return mut_pop


def uniformIntMutation(pop, mut_prob, ind_prob, bounds, debug=False):
    '''
    Peforms uniform integer mutation on integer populations.
    
    Parameters:
    pop (numpy array, required): The population, individuals as rows
    mut_prob (real between 0 and 1, required): The probability that any individual will mutate
    ind_prob (real between 0 and 1, required): The probability that a gene will mutate
    bounds (list, required): the [lower, upper] bounds of possible values for each gene   
    debug (boolean, optional, default=False): Flag to indicate whether to output debugging print statements.
    
    Returns: 
    Population of integer variables
    '''
    mut_pop = pop.copy()
    pop_size, ind_size = pop.shape[0], pop.shape[1]
    for j in range(pop_size):
        if np.random.uniform()<mut_prob:
            new_assign = mut_pop[j].copy()
            for i in range(ind_size):
                if np.random.uniform() < ind_prob:
                    if debug:
                        print('Gene', i, 'in Individual', j, 'mutated.')
                    while new_assign[i] == mut_pop[j][i]: # loops until new and old are different
                        new_assign[i] = np.random.randint(bounds[0], bounds[1])                     
            mut_pop[j] = new_assign
    return mut_pop.astype(int)


def bitFlipMutation(pop, mut_prob, ind_prob, debug=False):
    '''
    Peforms bit-flipping mutation on boolean populations.
    
    Parameters:
    pop (numpy array, required): The population, individuals as rows
    mut_prob (real between 0 and 1, required): The probability that any individual will mutate
    ind_prob (real between 0 and 1, required): The probability that a gene will mutate
    debug (boolean, optional, default=False): Flag to indicate whether to output debugging print statements.
    
    Returns: 
    Population of boolean variables
    '''
    
    #get sizes
    pop_size, ind_size = pop.shape[0], pop.shape[1]
    mut_pop = np.zeros((pop_size,ind_size),dtype=bool) # initialize mutation population
    for j in range(pop_size):
        individual = pop[j].copy() # copy is necessary to avoid conflicts in memory
        if np.random.uniform()<mut_prob:
            for i in range(ind_size):
                if np.random.uniform() < ind_prob:
                    if debug:
                        print('Gene', i, 'in Individual', j, 'flipped.')
                    individual[i] = ~individual[i]
        mut_pop[j] = individual.copy() # copy is necessary to avoid conflicts in memory
    return mut_pop.astype(bool)

def shuffleMutation(pop, mut_prob, ind_prob, debug=False):
    '''
    Peforms index shuffling mutation on permutation populations.
    
    Parameters:
    pop (numpy array, required): The population, individuals as rows
    mut_prob (real between 0 and 1, required): The probability that any individual will mutate
    ind_prob (real between 0 and 1, required): The probability that a gene will mutate
    debug (boolean, optional, default=False): Flag to indicate whether to output debugging print statements.
    
    Returns: 
    Population of permutation integer variables
    '''
    #get sizes
    pop_size, ind_size = pop.shape[0], pop.shape[1]
    if debug:
        print('Pop size:', pop_size)
        print('Individual size:', ind_size)
    mut_pop = np.zeros((pop_size,ind_size),dtype=int) # initialize mutation population
    for j in range(pop_size):
        individual = pop[j].copy() # copy is necessary to avoid conflicts in memory
        if np.random.uniform()<mut_prob:
            for k in range(ind_size):
                if np.random.uniform() < ind_prob:
                    swap = np.random.randint(ind_size)
                    if debug:
                        print('Gene', k, 'in Individual', j, 'swapped with', swap)
                        print('Individual before\n', individual)
                    individual[k],individual[swap] = individual[swap],individual[k]
                    if debug:
                        print('Individual after\n', individual)
        mut_pop[j] = individual.copy() # copy is necessary to avoid conflicts in memory
    return mut_pop.astype(int)     


# Elitism
def addElitism(initPop, mutPop, num_elite):
    '''
    Peforms elitism by pulling in num_elite best individuals from initPop to mutPop.
    
    Parameters:
    initPop (numpy array, required): The population from the beginning of the loop, individuals as rows
    mutPop (numpy array, required): The population post-mutation.
    
    Returns: 
    Population numpy array population
    '''    
    pop_size = initPop.shape[0]
    initPop[(num_elite):pop_size] = mutPop[(num_elite):pop_size].copy()
    return initPop

###############################
# Stats Tracking
###############################

def initStats(fitness, pop, num_iter):
    '''
    Sets up initial stats tracking
    
    Parameters:
    fitness (numpy array, required): The current fitness at the start of the algorithm
    pop (numpy array, required): The population for which we are tracking fitness
    num_iter (integer, required): The number of iterations we'll track.
    
    Returns: 
    stats (numpy array)
    best_fitness (real)
    best_x (individual)
    '''
    stats = np.zeros((num_iter+1,3)) # for collecting statistics
    #get the initial best fitness
    best_fitness = min(fitness)
    min_fitness = min(fitness) # best for this iteration
    index = np.argmin(fitness)
    #set the initial best_x
    best_x = pop[index]
    # initialize stats and output
    stats[0,:] = np.array([0,best_fitness, best_fitness])
    print('Iteration | Best this iter |    Best ever')
    return stats, best_fitness, best_x


def updateStats(stats, fitness, best_x, pop, iter, update_iter):
    '''
    Updates stats tracking
    
    Parameters:
    stats (numpy array, required): The stats that have been collected so far
    fitness (numpy array, required): The current fitness at the start of the algorithm
    best_x (numpy array individual, required): The current best individual
    pop (numpy array, required): The population for which we are tracking fitness
    iter (integer, required): The current iteration we are on.
    update_iter (integer, required): How often to display stats
    
    Returns: 
    stats (numpy array)
    best_fitness (real)
    best_x (individual)
    '''
    # collect stats and output to screen
    min_fitness = min(fitness) # best for this iteration

    #get stats less than this iteration
    snipped_stats = stats[0:iter]
    if len(snipped_stats) > 0:
        index = np.argmin(snipped_stats[:,2])
        best_fitness = min(snipped_stats[:,1])
    else:
        best_fitness = min_fitness
        best_x = []
    if min_fitness < best_fitness: # best for all iterations
        best_fitness = min_fitness
        index = np.argmin(fitness)
        best_x = pop[index]

    stats[iter+1,:] = np.array([iter+1,min_fitness, best_fitness])
    if (iter+1) % update_iter == 0 or (iter+1) ==1 :
        print(f"{stats[iter+1,0]:9.0f} | {stats[iter+1,1]:14.3e} | {stats[iter+1,2]:12.3e}")
    
    return stats, best_fitness, best_x


