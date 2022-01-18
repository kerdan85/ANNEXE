import numpy
import random

#GENETIC ALGORITHM OPERATORS

#Mating function
def mating_pool(pop_inputs, objective, num_parents):
    
    objective = numpy.asarray(objective)
    parents = [[None,None,None, None, None, None, None, None, None, None,
                None,None,None, None, None, None, None, None, None, None]]* num_parents
    for parent_num in range(num_parents):
        best_fit_index = numpy.where(objective == numpy.max(objective))
        best_fit_index = best_fit_index[0][0]
        parents[parent_num] = pop_inputs[best_fit_index, :]
        objective[best_fit_index] = -9999999
    return parents

#Crossover function
def crossover(parents, offspring_size):
    
    offspring = [[None,None,None, None, None, None, None, None, None, None,
                None,None,None, None, None, None, None, None, None, None]]* offspring_size[0]
    crossover_loc = numpy.uint32(offspring_size[1]/2)
    parents_list = parents.tolist()
    for k in range(offspring_size[0]):
        # Loc first parent
        parent_1_index = k%parents.shape[0]
        # Loc second parent
        parent_2_index = (k+1)%parents.shape[0]
        # Offspring generation
        offspring[k] = parents_list[parent_1_index][0:crossover_loc] + parents_list[parent_2_index][crossover_loc:]
    return offspring

def mutation(offspring_crossover, sol_per_pop, num_parents_mating, mutation_percent):

    offspring_crossover_a = numpy.asarray(offspring_crossover) # convert to array to do shape calculations
    num_mutations = numpy.uint32((mutation_percent*offspring_crossover_a.shape[1])/100)
    mutation_indices = numpy.array(random.sample(range(0, offspring_crossover_a.shape[1]), num_mutations))
    offspring_mutation = offspring_crossover * sol_per_pop
    offspring_mutation = offspring_mutation [:sol_per_pop-offspring_crossover_a.shape[0]]
    offspring_mutation = numpy.asarray(offspring_mutation, dtype=object)

    for index in range(sol_per_pop-int(num_parents_mating/2)):
            
        if 0 in mutation_indices:
            value = random.uniform(20,25)
            offspring_mutation[index, 0] = value
            
        if 1 in mutation_indices:
            value = random.randint(0,360)
            offspring_mutation[index, 1] = value
            
    
        if 2 in mutation_indices:
            value = random.uniform(0.001,2)
            offspring_mutation[index, 2] = value

        if 3 in mutation_indices:
            value = random.uniform(0.001,2)
            offspring_mutation[index, 3] = value
            
        if 4 in mutation_indices:
            value = random.uniform(0.001,2)
            offspring_mutation[index, 4] = value

        if 5 in mutation_indices:
            value = random.uniform(0.001,2)
            offspring_mutation[index, 5] = value
            
        if 6 in mutation_indices:
            value = random.uniform(0.001,2)
            offspring_mutation[index, 6] = value
        
        if 7 in mutation_indices:
            value = random.uniform(0.001,2)
            offspring_mutation[index, 7] = value
            
        if 8 in mutation_indices:
            value = random.uniform(0.001,2)
            offspring_mutation[index, 8] = value
            
        if 9 in mutation_indices:
            value = random.uniform(0.001,2)
            offspring_mutation[index, 9] = value
            
        if 10 in mutation_indices:
            value = random.uniform(0.001,2)
            offspring_mutation[index, 10] = value

        if 11 in mutation_indices:
            glazing = [0]*12 + [1]*1
            random.shuffle(glazing)
            value = glazing
            offspring_mutation[index, 11] = value
            
        if 12 in mutation_indices:
            hvac = [0]*6 + [1]*1
            random.shuffle(hvac)
            value = hvac
            offspring_mutation[index, 12] = value

        if 13 in mutation_indices:
            wall = [0]*9 + [1]*1
            random.shuffle(wall)
            if wall[0] == 1:
                wall.append(0)
            elif wall[1] == 1:
                wall.append(random.uniform(2,16))
            elif wall[2] == 1:
                wall.append(random.uniform(2,15))
            elif wall[3] == 1:
                wall.append(random.uniform(2,15))
            elif wall[4] == 1:
                wall.append(random.uniform(2,18))
            elif wall[5] == 1:
                wall.append(random.uniform(6.5,10))
            elif wall[6] == 1:
                wall.append(random.uniform(2,30))
            elif wall[7] == 1:
                wall.append(random.uniform(2,10))
            elif wall[8] == 1:
                wall.append(random.uniform(0.5,4))
            elif wall[9] == 1:
                wall.append(random.uniform(1,2))   
            value = wall
            offspring_mutation[index, 13] = value
            
        if 14 in mutation_indices:
            roof = [0]*9 + [1]*1
            random.shuffle(roof)
            if roof[0] == 1:
                roof.append(0)
            elif roof[1] == 1:
                roof.append(random.uniform(2,16))
            elif roof[2] == 1:
                roof.append(random.uniform(2,15))
            elif roof[3] == 1:
                roof.append(random.uniform(2,15))
            elif roof[4] == 1:
                roof.append(random.uniform(2,18))
            elif roof[5] == 1:
                roof.append(random.uniform(6.5,10))
            elif roof[6] == 1:
                roof.append(random.uniform(2,30))
            elif roof[7] == 1:
                roof.append(random.uniform(2,10))
            elif roof[8] == 1:
                roof.append(random.uniform(0.5,4))
            elif roof[9] == 1:
                roof.append(random.uniform(1,2))   
            value = roof
            offspring_mutation[index, 14] = value
        
        if 15 in mutation_indices:
            lights = [0]*1 + [1]*1
            random.shuffle(lights)
            value = lights
            offspring_mutation[index, 15] = value
        
        if 16 in mutation_indices:
            value = random.uniform(0,100)
            offspring_mutation[index, 16] = value
            
        if 17 in mutation_indices:
            wind = [0]*2 + [1]*1
            random.shuffle(wind)
            value = wind
            offspring_mutation[index, 17] = value

        if 18 in mutation_indices:
            value = random.uniform(0,50) 
            offspring_mutation[index, 18] = value
            
        if 19 in mutation_indices:
            value = random.uniform(0.001,0.15) 
            offspring_mutation[index, 19] = value
        
            
            
     
    return offspring_mutation
