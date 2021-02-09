import numpy as np
import pandas as pd
import random as rd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def initialization(size, n_feat):
    population = []
    sample_arr = [True, False]
    for i in range(size):
        chromosome = np.random.choice(sample_arr, size = n_feat)
        population.append(chromosome)
    return population



def find_parents_ts(population, X_train, X_test, y_train, y_test):
    
    parents = np.empty((0,np.size(population,1)), dtype = np.bool)
    
    for i in range(2):
        
        indices_list = np.random.choice(len(population),3,replace=False)
        
        posb_parent_1 = population[indices_list[0]]
        posb_parent_2 = population[indices_list[1]]
        posb_parent_3 = population[indices_list[2]]
        
        obj_func_parent_1 = fitting(posb_parent_1, X_train, X_test, y_train, y_test)
        obj_func_parent_2 = fitting(posb_parent_2, X_train, X_test, y_train, y_test)
        obj_func_parent_3 = fitting(posb_parent_3, X_train, X_test, y_train, y_test)
        
        min_obj_func = min(obj_func_parent_1,obj_func_parent_2,obj_func_parent_3)
        
        if min_obj_func == obj_func_parent_1:
            selected_parent = posb_parent_1
        elif min_obj_func == obj_func_parent_2:
            selected_parent = posb_parent_2
        else:
            selected_parent = posb_parent_3
        
        parents = np.vstack((parents,selected_parent))
        
    parent_1 = parents[0,:]
    parent_2 = parents[1,:]
    
    return parent_1,parent_2


def fitting(chromosome, X_train, X_test, y_train, y_test):
    scores = []
    logmodel = LogisticRegression()
    logmodel.fit(X_train.iloc[:, chromosome], y_train)
    predictions = logmodel.predict(X_test.iloc[:, chromosome])
    scores.append(accuracy_score(y_test, predictions))
    return scores



def crossover(parents, n_feat, prob_crossover=1):
    random_prob_crossover = np.random.rand()
    child_1 = np.empty((0, len(parents[0])), dtype = bool)
    child_2 = np.empty((0, len(parents[1])), dtype = bool)
    
    if random_prob_crossover < prob_crossover:
        parent_1 = parents[0]
        parent_2 = parents[1]

        rand = np.sort(rd.sample(range(n_feat), 2))
        index_1 = rand[0]
        index_2 = rand[1]

        first_seg_parent_1 = parent_1[:index_1]
        mid_seg_parent_1 = parent_1[index_1:index_2]
        last_seg_parent_1 = parent_1[index_2:]

        first_seg_parent_2 = parent_2[:index_1]
        mid_seg_parent_2 = parent_2[index_1:index_2]
        last_seg_parent_2 = parent_2[index_2:]

        child_1 = np.concatenate((first_seg_parent_1, mid_seg_parent_2, last_seg_parent_1))
        child_2 = np.concatenate((first_seg_parent_2, mid_seg_parent_1, last_seg_parent_2))
    
    else:
        child_1 = parents[0]
        child_2 = parents[1]
    
    return child_1, child_2



def mutation(pop_after_cross, prob_mutation=.2):
    
    child_1 = pop_after_cross[0]
    child_2 = pop_after_cross[1]
    
    mutated_child_1 = np.empty((0, len(child_1)), dtype = bool)
    mutated_child_2 = np.empty((0, len(child_2)), dtype = bool)
    
    t = 0
    for i in child_1:
        rand_num_mutation = np.random.rand()

        if rand_num_mutation < prob_mutation:

            if child_1[t] == True:
                child_1[t] = False
            else:
                child_1[t] = True

            mutated_child_1 = child_1
            t = t + 1

        else:
            mutated_child_1 = child_1
            t = t + 1
            
    t = 0
    for i in child_2:
        rand_num_mutation = np.random.rand()

        if rand_num_mutation < prob_mutation:

            if child_2[t] == True:
                child_2[t] = False
            else:
                child_2[t] = True
                
            mutated_child_2 = child_2
            t = t + 1

        else:
            mutated_child_2 = child_2
            t = t + 1
            
    return mutated_child_1, mutated_child_2



def generation(size_of_population, n_feat, generation, X_train, X_test, y_train, y_test):
    start_time = time.time()
    gen = 1
    stats = pd.DataFrame()
    best_chromosome = pd.DataFrame()
    population = initialization(size_of_population, n_feat)
    

    for i in range(generation):

        print()
        print('--> Generation: #,',gen)

        family = 1
        obj_val_total = []
        new_population = np.empty((0,n_feat), bool)


        for j in range(int(len(population)/2)):
            print()
            print('--> Family: #,', family)

            parent_1 = find_parents_ts(population, X_train, X_test, y_train, y_test)[0]
            parent_2 = find_parents_ts(population, X_train, X_test, y_train, y_test)[1]

            parents = np.vstack((parent_1, parent_2))

            child_1 = crossover(parents, n_feat)[0]
            child_2 = crossover(parents, n_feat)[1]

            children = np.vstack((child_1, child_2))

            mutated_child_1 = mutation(children)[0]
            mutated_child_2 = mutation(children)[1]

            obj_val_mutated_child_1 = fitting(mutated_child_1, X_train, X_test, y_train, y_test)
            obj_val_mutated_child_2 = fitting(mutated_child_2, X_train, X_test, y_train, y_test)

            print()
            print('Obj Val for Mutated Child #1 at Generation #{} : {}'.
                 format(gen, obj_val_mutated_child_1))
            print('Obj Val for Mutated Child #2 at Generation #{} : {}'.
                 format(gen, obj_val_mutated_child_2))


            temp = np.append(obj_val_mutated_child_1, mutated_child_1)
            temp = pd.Series(temp)
            best_chromosome = best_chromosome.append(temp, ignore_index=True)

            temp = np.append(obj_val_mutated_child_2, mutated_child_2)
            temp = pd.Series(temp)
            best_chromosome = best_chromosome.append(temp, ignore_index=True)

            obj_val_total.append(obj_val_mutated_child_1)
            obj_val_total.append(obj_val_mutated_child_2)

            new_population = np.append(new_population, [mutated_child_1], axis=0)
            new_population = np.append(new_population, [mutated_child_2], axis=0)

            family = family + 1



        population = new_population

        mean_of_gen = np.mean(obj_val_total)
        var_of_gen = np.var(obj_val_total)
        max_of_gen = np.max(obj_val_total)
        min_of_gen = np.min(obj_val_total)

        series = pd.Series([gen, mean_of_gen, var_of_gen, max_of_gen, min_of_gen])
        stats = stats.append(series, ignore_index=True)

        gen = gen+1
        
    return best_chromosome, stats

    end_time = time.time()
    running_time = end_time - start_time
    print(running_time)