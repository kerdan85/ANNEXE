import tensorflow as tf
import os
import numpy
import genetic_2nd
import ann
import csv
import numpy as np
import random
import data_input
import gc
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, r2_score


#Parallel units definition
NUM_PARALLEL_EXEC_UNITS = 4

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = NUM_PARALLEL_EXEC_UNITS, 
         inter_op_parallelism_threads = 2, 
         allow_soft_placement = True, 
         device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"


# load dataset   
dataframe = pd.read_csv("Data.csv", sep=',', header = None)
dataset = dataframe.values
    
    # split into input (X) and output (Y) variables, splitting csv data
X = dataset[:,0:61]
   
Y = dataset[:,-3:]
    
    # split X, Y into a train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42)
    #X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=42)
    
    # created scaler
scaler = StandardScaler()
    # fit scaler on training dataset
scaler.fit(Y_train)
    # transform training dataset
Y_train = scaler.transform(Y_train)
    # transform test dataset
Y_test = scaler.transform(Y_test)

    
model = Sequential()

model.add(Dense(20, input_dim=61, kernel_initializer='uniform', activation='tanh'))
#model.add(Dense(20, kernel_initializer='normal', activation='elu'))
#model.add(Dense(20, kernel_initializer='uniform', activation='elu'))
model.add(Dense(3, kernel_initializer='uniform',activation='linear'))


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
history = model.fit(X_train, Y_train, epochs=200, batch_size=25, verbose=1, validation_data=(X_test, Y_test))
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title("ANN Performance \n 1 Hidden layer [20,] ", fontsize=12, fontweight=0, color='black')
pyplot.xlabel('Epochs')
pyplot.ylabel('Mean Squared Error (mse)')
pyplot.legend(('training   [95%]', 'testing    [5%]'), loc='best')

pyplot.savefig('.\Figures\ANN_performance_MSE.pdf') 
pyplot.show()

import math
rmse = []

for x in history.history['loss']:
    rmse.append(math.sqrt(x))
    
rmse_v = []
for x in history.history['val_loss']:
    rmse_v.append(math.sqrt(x))
    
pyplot.plot(rmse)
pyplot.plot(rmse_v)
pyplot.title("ANN Performance \n 1 Hidden layer [20,] ", fontsize=12, fontweight=0, color='black')
pyplot.xlabel('Epochs')
pyplot.ylabel('Root Mean Squared Error (rmse)')
pyplot.legend(('training   [95%]', 'testing    [5%]'), loc='best')
pyplot.savefig('.\Figures\ANN_performance_RMSE.pdf') 
pyplot.show()

# =============================================================================
#     '''The result reports the mean squared error including the average and 
#     standard deviation (average variance) across all 10 folds of the cross validation evaluation.'''
# =============================================================================


predictions = model.predict(X_train)


MSE_scaled = mean_squared_error(Y_train, predictions)
R2 = r2_score(Y_train, predictions)
print("Results-- MSE: %.2f R2: %.2f " % (MSE_scaled, R2))
    
          

#Genetic Algorithm Parameters

num_generations = 100
sol_per_pop = 100
num_parents_mating = 20
mutation_percent = 20



#Creating an empty list to store the initial population
initial_population = []

#Creating an empty list to store the final solutions
final_list=[]

#Create initial population
for curr_sol in numpy.arange(0, sol_per_pop):
    
    initial_population_ = []
    
    setpoint = random.uniform(20,25)
    orientation = random.randint(0,360)
    over_south = random.uniform(0.001,2)
    over_north = random.uniform(0.001,2)
    over_west = random.uniform(0.001,2)
    fin_south_right = random.uniform(0.001,2)
    fin_north_right = random.uniform(0.001,2)
    fin_west_right = random.uniform(0.001,2)
    fin_south_left = random.uniform(0.001,2)
    fin_north_left = random.uniform(0.001,2)
    fin_west_left = random.uniform(0.001,2)
    
    glazing = [0]*12 + [1]*1
    random.shuffle(glazing)
    
    
    hvac = [0]*6 + [1]*1
    random.shuffle(hvac)
    
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
        
    lights = [0]*1 + [1]*1
    random.shuffle(lights)
    
    pv = random.uniform(0,100)
    
    wind = [0]*2 + [1]*1
    random.shuffle(wind)
        
    swh = random.uniform(0,50) 
    
    air_gap = random.uniform(0.001,0.15) 
    
    
        
    initial_population.append([setpoint, orientation, over_south, over_north, over_west, fin_south_right, fin_north_right,
                               fin_west_right,fin_south_left, fin_north_left,fin_west_left,
                                glazing, hvac, wall, roof, lights, pv, wind, swh, air_gap ])
    


#Initial population
pop_inputs = np.asarray(initial_population) 
del(initial_population)

#Start GA process
for generation in range(num_generations):    
    pre_list=[]
    list_inputs =[]
    #list_fitness=[]
    list_objective=[]
    list_other_metrics = []
  
    
    print("================================================================")
    print("================================================================")
    print("\nGeneration : ", generation+1)
    print("Inputs : \n",  pop_inputs)

    pop_inputs = pop_inputs      
                             
    # Measuring the fitness of each solution in the population.
    #fitness = []
    objective = []
    other_metrics =[]
    
   
    
    
    #ANN model training for sol_population p in generation g
    for index in range(sol_per_pop):
        
        print('\n Generation: ', generation+1, " of ", num_generations, ' Simulation: ', index+1 ,' of ', sol_per_pop)
        
        
        a = []        
        for i in range(len(pop_inputs[index])):
            
            if type(pop_inputs[index][i]) == list:
                b = np.asarray(pop_inputs[index][i])
                for x in b:
                    a.append(x)
            else:        
                a.append(pop_inputs[index][i])
            
        a_ = np.asarray([a], dtype=np.object).astype('float32')
        test = model.predict(a_)
        unscaled = scaler.inverse_transform(test[0])
        
        
        
        #OBJECTIVE FUNCTION
        obj = -(unscaled[0]/max(dataset[:,65])+unscaled[1]+
                    unscaled[2]/max(dataset[:, 67]))/3
            
            #Appending obj 1 and 2
           # fitness.append([RMSE, RMSE_val])
            #fitness.append([obj])
            
            #Appending objective list
        objective.append([obj])
        
        print("Objective")
        print(obj)
        other_metrics.append([unscaled[0], unscaled[1], unscaled[2]])
           # del  X_train, Y_train, X_test, Y_test
        gc.collect()
    
   # print(fitness)
    print(objective)
    
   # list_fitness.append(fitness)
    list_objective.append(objective)
    list_inputs.append(pop_inputs.tolist())
    list_other_metrics.append(other_metrics)
    
    
    # top performance ANN model in the population are selected for mating.
    parents = genetic_2nd.mating_pool(pop_inputs, 
                                    objective.copy(), 
                                    num_parents_mating)
    print("Parents")
    print(parents)
    parents = numpy.asarray(parents) 


    # Crossover to generate the next geenration of solutions
    offspring_crossover = genetic_2nd.crossover(parents,
                                       offspring_size=(int(num_parents_mating/2), pop_inputs.shape[1]))
    print("Crossover")
    print(offspring_crossover)


    # Mutation for population variation
    offspring_mutation = genetic_2nd.mutation(offspring_crossover, sol_per_pop, num_parents_mating, 
                                     mutation_percent=mutation_percent)
    
    print("Mutation")
    print(offspring_mutation)
        
    # New population for generation g+1
    pop_inputs[0:len(offspring_crossover), :] = offspring_crossover
    pop_inputs[len(offspring_crossover):, :] = offspring_mutation
    print('NEW INPUTS :\n', pop_inputs )
    
    
    for x in range(len(list_inputs)):
        for y in range(len(list_inputs[0])):
            pre_list = list_inputs[x][y]
            pre_list.append(list_objective[x][y][0])
            for w in range(len(list_other_metrics[x][y])):
                pre_list.append(list_other_metrics[x][y][w])
                      
            final_list.append(pre_list)      
   
    del(objective, parents, offspring_mutation, offspring_crossover, list_inputs,  list_objective, pre_list)
    gc.collect()
    
#Insert headers to final list
final_list.insert(0, ['setpoint', 'orientation', 'over_south', 'over_north', 'over_west', 
                      'fin_south_right', 'fin_north_right', 'fin_west_right','fin_south_left',
                      'fin_north_left','fin_west_left', 'glazing', 'hvac', 'wall', 'roof', 
                      'lights', 'pv', 'wind', 'swh', 'air_gap', 'objective', 'Exergy_dest', 'discomfort', 'LCA'])
final_list
 
#Saving all ANN structures, hyperparameters and metrics
with open('FINAL_RESULTS_2nd_stage.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerows(final_list)
        


