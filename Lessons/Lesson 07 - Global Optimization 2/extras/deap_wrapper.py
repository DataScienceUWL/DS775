import importlib

def genetic_algorithm(fitness,ind_type,ind_size, pop_size, cx_prob, mut_prob, max_gen, max_no_improve, lower = None, upper = None, minimize=True, random_seed = None, num_proc = 1, user_config_dict=None, loc_search_fun=None, ls_update_type = 'best', ls_num_update = 1, **kwargs):

    '''for single objective optimization, based on the DEAP package

    fitness:  objective function should take an array (an individual from the population) and produce a number

    ind_type:  each individual is an one dimensional array with ind_size entries
        'float':  floating point, must supply lower and upper bounds
        'integer':  integer, must supply lower and upper bounds
        'boolean':  boolean
        'permutation':  individual is a permutation of the sequence 0,1,...,ind_size-1

    ind_size:  length or dimension of each individual

    pop_size:  number of individuals in population, larger for a more thorough and slower search

    cx_prob:  probability that a selected pair of individuals will mate, between 0 and 1

    mut_prob:  probability that a selected idividual will mutate

    max_gen:  maximum number of iterations (generations) to evolve population

    max_no_improve:  maximum number of iterations to evolve without improvement (set to max_gen to run all max_gen iterations)
    
    lower:  lower bound for float and integer types
    
    upper:  upper bound for float and integer types

    minimize:  defaults to True to minimize objective, set to False to maximize

    random_seed:  set to get reproducible results

    num_proc:  set > 1 to use multiple processes for fitness evaluation (EXPERIMENTAL, requires multiprocessing package)

    user_config_dict:  can use this to choose different types of crossover and mutation, without it defaults are used
    
    loc_search_fun:  must accept an individual x and return minimizer x,y, pass additional arguments with to the objective function with **kwargs, if None then no local search, could also be used to modify individuals (mutate, sort, whatever)
    
    

    **kwargs:  additional arguments that are passed directly to fitness function'''

    # set DEAP weights for optimization
    if minimize:
        weights = (-1.0,)
    else:
        weights = (1.0,)

    # set default evolution parameters for each time as well as sampling method for init
    if ind_type == 'permutation':
        def create_individual(pcls,ind_size,ls):
            ind = pcls(loc_random.sample(range(ind_size),ind_size))
            ind.loc_searched = ls
            return(ind)
        config_dict = {"select_op":"selTournament", "select_param":{"tournsize":3},
                          "mate_op":"cxOrdered", "mate_param":{},
                          "mute_op":"mutShuffleIndexes", "mute_param":{"indpb":0.05}}
    elif ind_type == 'float':
        def create_individual(pcls,ind_size,ls):
            ind = pcls([loc_random.uniform(lower,upper) for i in range(ind_size)])
            ind.loc_searched = ls
            return(ind)
        config_dict = {"select_op":"selTournament", "select_param":{"tournsize":3},
                          "mate_op":"cxBlend","mate_param":{"alpha":.2},
                          "mute_op":"mutGaussian", "mute_param":{"mu":0,"sigma":1,"indpb":0.05}}
    elif ind_type == 'boolean':
        def create_individual(pcls,ind_size,ls):
            ind = pcls(loc_random.choices([True, False], k=ind_size))
            ind.loc_searched = ls
            return ind
        config_dict = {"select_op":"selTournament", "select_param":{"tournsize":3},
                          "mate_op":"cxOnePoint", "mate_param":{},
                          "mute_op":"mutFlipBit", "mute_param":{"indpb":0.05}}
    elif ind_type == 'integer':
        def create_individual(pcls,ind_size,ls):
            # a little hacky to get lower and upper from outside function
            ind = pcls([loc_random.randint(lower,upper) for i in range(ind_size)])
            ind.loc_searched = ls
            return ind
            
        config_dict = {"select_op":"selTournament", "select_param":{"tournsize":3},
                          "mate_op":"cxOnePoint", "mate_param":{},
                          "mute_op":"mutUniformInt", "mute_param":{"indpb":0.05,"low":lower,"up":upper}}
        
    # include user options in configuration dictionary (doesn't catch errors!)
    if user_config_dict is not None:
        for key in config_dict.keys():
            if user_config_dict.get(key) is not None:
                config_dict[key] = user_config_dict[key]
        
    # return result as tuple with element for DEAP
    def fitness_tuple(individual,**kwargs):
        return (fitness(individual,**kwargs),)
    
    # create local copies of each module
    loc_random = importlib.import_module("random")
    loc_creator = importlib.import_module("deap.creator")
    loc_algorithms = importlib.import_module("deap.algorithms")
    loc_base = importlib.import_module("deap.base")
    loc_tools = importlib.import_module("deap.tools")
    loc_numpy = importlib.import_module("numpy")

    # create problem
    loc_creator.create("optFitness", loc_base.Fitness, weights=weights)
    loc_creator.create("Individual", list, fitness=loc_creator.optFitness)
    
    # setup initialization, population, and evolution operators
    loc_toolbox = loc_base.Toolbox()
    loc_toolbox.register("individual",create_individual,loc_creator.Individual,ind_size,False)
    loc_toolbox.register("population", loc_tools.initRepeat, list, loc_toolbox.individual)
    loc_toolbox.register("evaluate", fitness_tuple, **kwargs)

    loc_toolbox.register("select",eval("loc_tools."+config_dict["select_op"]),
                         **config_dict["select_param"])
    loc_toolbox.register("mate", eval("loc_tools."+config_dict["mate_op"]),
                         **config_dict["mate_param"])
    loc_toolbox.register("mutate", eval("loc_tools."+config_dict["mute_op"]),
                         **config_dict["mute_param"])
    
    # add decorator to mutation to clip bounds for ind_type float
    if ind_type == "float":
        def checkBounds(min, max):
            def decorator(func):
                def wrapper(*args, **kargs):
                    offspring = func(*args, **kargs)
                    for child in offspring:
                        for i in range(len(child)):
                            if child[i] > max:
                                child[i] = max
                            elif child[i] < min:
                                child[i] = min
                    return offspring
                return wrapper
            return decorator
        loc_toolbox.decorate("mutate", checkBounds(lower,upper))

    loc_stats = loc_tools.Statistics(lambda ind: ind.fitness.values)
    loc_stats.register("avg", loc_numpy.mean)
    loc_stats.register("std", loc_numpy.std)
    loc_stats.register("min", loc_numpy.min)
    loc_stats.register("max", loc_numpy.max)
    
    loc_random.seed(random_seed)

    pop = loc_toolbox.population(n=pop_size)
    hof = loc_tools.HallOfFame(1)
    logbook = loc_tools.Logbook()
    loc_search_count = 0

    # Evaluate the entire population
    fitnesses = list(map(loc_toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        ind.loc_searched = False

    hof.update(pop)
    best_val = hof[0].fitness.values
    num_no_improve = 0
    generation = 0
    
    while num_no_improve < max_no_improve and generation < max_gen:

        # do local searches for individuals in population (before selection)
        if loc_search_fun is not None:
            # find indices to locally search according to update strategy
            if ls_update_type == 'best':
                pop.sort(key=lambda x:x.fitness.values,reverse=not minimize)
            elif ls_update_type == 'worst':
                pop.sort(key=lambda x:x.fitness.values,reverse=minimize)
            elif ls_update_type == 'random':
                random_idx = loc_random.sample(range(pop_size),pop_size)
                pop = [pop[i] for i in random_idx]
            # do the local search for the first ls_num_update individuals in the population
            loc_search_count = 0
            for i in range(ls_num_update):
                if not pop[i].loc_searched:
                    loc_search_count += 1
                    best_x, best_y = loc_search_fun(pop[i],**kwargs)
                    pop[i] = loc_creator.Individual(best_x)
                    pop[i].fitness.values = (best_y,)
                    pop[i].loc_searched = True

        # Select the next generation individuals
        selected = loc_toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(loc_toolbox.clone, selected))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if loc_random.random() < cx_prob:
                loc_toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                child1.loc_searched=False
                child2.loc_searched=False

        for mutant in offspring:
            if loc_random.random() < mut_prob:
                loc_toolbox.mutate(mutant)
                del mutant.fitness.values
                mutant.loc_searched=False

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(loc_toolbox.evaluate, invalid_ind)
        num_evals = 0
        for ind, fit in zip(invalid_ind, fitnesses):
            num_evals += 1
            ind.fitness.values = fit
            ind.loc_searched = False
            

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # track the best value and reset counter if there is a change
        hof.update(pop)
        curr_best_val = hof[0].fitness.values[0]
        num_no_improve += 1
        if curr_best_val != best_val:
            best_val = curr_best_val
            num_no_improve = 0

        # record stats
        record = loc_stats.compile(pop)
        logbook.record(gen=generation, evals=num_evals, ls = loc_search_count, **record)

        # increment generation
        generation += 1

    best_x = list(hof[0])
    
    del loc_random, loc_creator, loc_algorithms, loc_base, loc_tools, loc_numpy

    return best_val, best_x, logbook