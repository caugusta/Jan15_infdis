#JUST neighbourhood epidemics

import numpy as np
from scipy import spatial
import json
import itertools
from utils import *
#from 30Decreadcount import *
MAX_FAILED_ATTEMPS = 10
import csv

def write_neighbourhood_epidemics(susc, trans, inf_period, eps, repetitions, pop):
        epi_list = []
        count_list = []
        new_susc = []
        new_trans = []
        new_inf_period = []
        new_eps = []
        count = 0

	parameters_product = itertools.product(trans, inf_period, eps)
        for transmissibility, infectious_period, epsilon in parameters_product:
                while True:
                    for rep in range(repetitions):
                        print 'neighbourhood', transmissibility, infectious_period, epsilon, rep, count
                        for _ in range(MAX_FAILED_ATTEMPS):
                            g1 = neigh_epidemic(
                                pop, susc, transmissibility,
                                infectious_period, epsilon, full_mat)
                            print 'g1', len(g1) #the number of people who became infectious
			    print abs(len(g1) - pop) < 2 #True means success
                            print 'max(g1.values)', max(g1.values()), infectious_period #the number of days the epidemic lasts
                            if abs(len(g1) - pop) < 2 and max(g1.values()) > infectious_period:
                                g2 = inf_per_count_time(g1)
                                print 'successful'
                                count += 1
                                epi_list.append(g1)
                                count_list.append(g2)
                                new_susc.append(susc)
                                new_trans.append(transmissibility)
                                new_inf_period.append(infectious_period)
                                new_eps.append(epsilon)
                                break
                        else:
			    transmissibility += 0.05
                            if rep > 0:
                                del epi_list[-rep:]
                                del count_list[-rep:]
                                del new_susc[-rep:]
                                del new_trans[-rep:]
                                del new_inf_period[-rep:]
                                del new_eps[-rep:]
                                count -=1
                            break #break out of repetitions
                    else:
                        break

        paras =  np.array([
                np.asarray(new_susc),
                np.asarray(new_trans),
                np.asarray(new_inf_period),
                np.asarray(new_eps)
        ]).T
        print 'number of parameter rows', paras[:,0].shape
        with open('parameters_neighbourhood_Jan14.txt', 'w') as newfile1:
                np.savetxt(newfile1, paras, fmt = ['%f', '%f', '%f', '%f'])


        with open('epidemics_neighbourhood_Jan14.json', 'w') as newfile2:
                json.dump(count_list, newfile2)

	with open('epidemics_individual_neighbourhood_Jan14.json', 'w') as newfile3:
		json.dump(epi_list, newfile3)

if __name__ == "__main__":
    pop = 50 #each cluster will have 100 individuals
    susc = 0.3
    pl_trans = [2.0, 2.2, 2.4, 2.6]
    inf_period = [5, 6]#[2, 3]
    eps = [0., 0.01, 0.02, 0.05] #was 0, 0.01, 0.02, 0.05
    reps = 400
    neigh_trans = [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]#, 1.6, 1.7, 1.8] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]# 1.1, 1.3, 1.5, 1.7]
    meanval = np.array([[1., 1.], [2., 2.], [3., 3.]])
    sigmaval = np.array([[[0.5, 0.], [0., 0.5]], [[0.5, 0.], [0., 0.5]], [[0.5, 0.], [0., 0.5]]])
    pop = pop*len(meanval) #total population

    ##Uncomment this to read in the population instead of creating one
    with open('14-3Jan_clustered_population.txt', 'r') as f:
           xy = np.loadtxt(f)
    x_pos = [item[0] for item in xy]
    y_pos = [item[1] for item in xy]
    dist_mat = np.asarray(zip(x_pos, y_pos))
    pdistance = scipy.spatial.distance.pdist(dist_mat)
    full_mat = scipy.spatial.distance.squareform(pdistance)

    write_neighbourhood_epidemics(susc, neigh_trans, inf_period, eps, reps, pop)