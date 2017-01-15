#Just postprocessing for pl_epidemics.py, which was started on Jan14

import numpy as np
import json
import itertools
from utils import *
import csv 
import re
import sys
import string
from operator import itemgetter
from itertools import tee, izip, chain, groupby
from functools import partial

#unnecessary
def check_consecutive():
        cc = []
        for k, g in groupby(enumerate(times), lambda(i, x): i-x):
                cc.append(map(itemgetter(1), g))
        return cc

#necessary
def pairwise(iterable):
	a, b = tee(iterable)
	next(b, None)
	return izip(a, b)

#necessary, from itertools.chain recipes
def flatten(listofLists):
	return chain.from_iterable(listofLists)

def read_pl_indivs(pop): #Clean
	with open( 'epidemics_individual_powerlaw_Jan14.json', 'r' ) as f :
                    data = f.read()

        count = len(data.split('}, {'))
        #count = len(data.split('}{')) #1600 as desired
        print 'powerlaw epidemics indiv', count

        str_epi = ' '.join(data.split('{'))
        str_epi2 = str_epi.split('}')


        times = re.findall(r'"(.\w*)"', str_epi2[0])
        infindivs = re.findall(r':(.\w*),', str_epi2[0]) #need this to also return the last element of the string
        lastone = str_epi2[0][-1]
        new1 = infindivs
        new1.append(lastone)

        times = np.asarray(times, dtype=np.int32)
        infindivs = np.asarray(infindivs, dtype=np.int32)
        newinf = infindivs

        i=1
        while i < len(str_epi2):
                times_1 = re.findall(r'"(.\w*)"', str_epi2[i])
                infindivs_1 = re.findall(r':(.\w*),', str_epi2[i])
                lastone = str_epi2[0][-1]
                newinf = infindivs_1
                newinf.append(lastone)
                if len(newinf) < 2:
                        print 'list was empty'
                        i += 1
                        continue
                else:
                        times = np.concatenate((times, np.asarray(times_1, dtype=np.int32)))
                        infindivs = np.concatenate((infindivs, np.asarray(infindivs_1, dtype=np.int32)))
                        i += 1

        starts = []
        starts.append(times[0]) #by default the start of an epidemic
        starts_index = []
        starts_index.append(0)
        count = 0
	for i, j in pairwise(times):
                count += 1
                if j < i:
                        #then we've started a new epidemic
                        starts.append(j)
                        starts_index.append(count)
                else:
                        continue

        ends = []
        ends_index = []
        for i in range(len(starts_index)):
                ends.append(times[starts_index[i]-1])
                ends_index.append(starts_index[i]-1)
        ends_index.pop(0)
        ends.pop(0)

	#print len(starts_index)
	#print len(ends_index)
	#print len(ends)
	#print len(starts)
	
        ends.append(times[-1])
        ends_index.append(-1)

        times_list_k = []
        infs_list_k = []

	#Too many epidemics - things are taking forever. Keep only the first 24 unique parameter settings.
	#starts_index = starts_index[:-3200]
        #ends_index = ends_index[:-3200]
	#ends = ends[:-3200]
	#starts = starts[:-3200]
	#print len(starts_index)
	#print len(ends_index)

	infs_array = np.zeros(10)	
        for i in range(len(starts_index)): #the number of epidemics
                #print 'i is', i
                a1 = set(np.arange(starts[i], ends[i]+1))
                times_list = list(times[starts_index[i]:ends_index[i]+1])
                infs_list = list(infindivs[starts_index[i]:ends_index[i]+1])
                t_list = np.array(times_list, dtype=np.int32) #http://stackoverflow.com/questions/21585807/fill-missing-values-in-python-array
                i_list = np.array(infs_list, dtype=np.int32)
                newinfs = np.empty((pop))
                newinfs.fill(-500)
                newinfs[t_list - 1] = i_list
                missing = np.nonzero(newinfs == -500)[0]
                newinfs[missing] = 0
	
		infs_array = np.concatenate([infs_array, newinfs])	
		a1 = np.arange(1, (pop+1))
		times_array = np.tile(a1, len(starts_index)) #number of epidemics
	infs_array = infs_array[10:]

	print times_array.shape
	print infs_array.shape

	with open('Jan14_powerlaw_epidemics_individual.txt', 'w') as f:
		np.savetxt(f, np.c_[times_array, infs_array], fmt=['%i', '%i'])

	return np.c_[times_array, infs_array]

def read_pl():

        with open( 'epidemics_powerlaw_Jan14.json', 'r' ) as f :
            data = f.read()

        count = len(data.split('}, {'))
        #count = len(data.split('}{')) #9600 as desired
        print 'powerlaw epidemics', count

        str_epi = ' '.join(data.split('{'))
        str_epi2 = str_epi.split('}')

        times = re.findall(r'"(.\w*)"', str_epi2[0])
        infindivs = re.findall(r':(.\w*),', str_epi2[0]) #need this to also return the last element of the string
        lastone = str_epi2[0][-1]
        new1 = infindivs
        new1.append(lastone)

	#Trying this
	#str_epi2 = str_epi2[:-3201]

        times = np.asarray(times, dtype=np.int32)
        infindivs = np.asarray(infindivs, dtype=np.int32)
        newinf = infindivs

        i=1
        while i < len(str_epi2):
                #print i
                times_1 = re.findall(r'"(.\w*)"', str_epi2[i])
                infindivs_1 = re.findall(r':(.\w*),', str_epi2[i])
                lastone = str_epi2[0][-1]
                newinf = infindivs_1
                newinf.append(lastone)
                if len(newinf) < 2:
                        print 'list was empty'
                        i += 1
                        continue
                else:
                        times = np.concatenate((times, np.asarray(times_1, dtype=np.int32)))
                        infindivs = np.concatenate((infindivs, np.asarray(infindivs_1, dtype=np.int32)))
                        i += 1

        with open('Jan14_powerlaw_epidemics.txt', 'w') as f:
                np.savetxt(f, np.c_[times, infindivs], fmt=['%i', '%i'])

        print 'wrote Jan14_powerlaw_epidemics.txt'
#Purpose: read in epidemics, write in an R-readable format
#and make it easier to process
def read_pl_prev():
        with open( 'epidemics_powerlaw_Jan14.json', 'r' ) as f :
            data = f.read()

        count = len(data.split('}, {'))
        #count = len(data.split('}{')) #9600 as desired
        print 'powerlaw epidemics', count

        str_epi = ' '.join(data.split('{'))
        str_epi2 = str_epi.split('}')

	#print len(str_epi2) #number of epidemics - 12800 (but is 12801...)
	#print len(str_epi2[:-3201]) #1 because of last ]
	#str_epi2 = str_epi2[:-3201]
	#sys.exit()

        times = re.findall(r'"(.\w*)"', str_epi2[0])
        infindivs = re.findall(r':(.\w*),', str_epi2[0]) #need this to also return the last element of the string
        lastone = str_epi2[0][-1]
        #new1 = infindivs
        #new1.append(lastone)

        times = np.asarray(times, dtype=np.int32)
        infindivs = np.asarray(infindivs, dtype=np.int32)
        newinf = infindivs

        i=1
        while i < len(str_epi2):
                #print i
                times_1 = re.findall(r'"(.\w*)"', str_epi2[i])
                infindivs_1 = re.findall(r':(.\w*),', str_epi2[i])
                lastone = str_epi2[0][-1]
                newinf = infindivs_1
                newinf.append(lastone)
                if len(newinf) < 2:
                        print 'list was empty'
                        i += 1
                        continue
                else:
                        times = np.concatenate((times, np.asarray(times_1, dtype=np.int32)))
                        infindivs = np.concatenate((infindivs, np.asarray(infindivs_1, dtype=np.int32)))
                        i += 1

        with open('Jan14_powerlaw_epidemics.txt', 'w') as f:
                np.savetxt(f, np.c_[times, infindivs], fmt=['%i', '%i'])

        print 'wrote Jan14_powerlaw_epidemics.txt'



#Purpose: read in the parameter list and the mean epidemic length associated with the 200
#epidemics that were run for that parameter setting, save as a file together
def pl_paras_and_means(paras, meanvec):
	#paras = paras[:24] #because only want first 24 epidemics
	#means = meanvec[:24] #again...
        eachline = map(lambda s: s.strip(), paras)
        listnew = [eachline[i].split() for i in range(len(eachline))]
        for i in range(len(listnew)):
                listnew[i].append(meanvec[i])
        listnew = [[float(i) for i in x] for x in listnew]
        with open('Jan14_pl_paras_and_means.csv', 'w') as newfile1:
                writer = csv.writer(newfile1)
                writer.writerows(listnew)
	return listnew

#Purpose: find the 1's in the epidemic lists.
#Useful to define the start (and end) of an epidemic
def find1(lst):
    result = []
    for i, x in enumerate(lst):
        if x is '1':
            result.append(i)
    return result

#Purpose: find the mean lengths of the epidemics
def find_means(filename, reps, ignore_list=[]): #"Dec30_powerlaw_epidemics.txt"
        f = open(filename, 'r')
        data = f.readlines()
        times = []
        for x in data:
                times.append(x.split(' ')[0]) #get first column. Is written as 'time #inf' so first column is time
        f.close()

        findones = find1(times) #there should be 6400 ones. These are the indices of the ones.
        endtimes = [times[findones[i]-1] for i in range(len(findones))]
        endtimes = map(np.int32, endtimes)

        mean_epi_length = []
        count_2 = 0

	print 'ignore_list', ignore_list
	#print len(ignore_list)
	#sys.exit()

	if len(ignore_list) > 0:
        	if str(count_2) not in ignore_list:
                	#print str(count_2), ignore_list
                	#print reps-1
			mean_epi_length.append(np.mean(endtimes[0:(reps-1)]))
			print mean_epi_length[0]
		for i in range(len(endtimes)): #original - worked for neighbourhood case, but I ignored 0:199 there
			if i % reps == 0: #for each epidemic set
				count_2 += 1
				print str(count_2), ignore_list
				print str(count_2) not in ignore_list
				print (reps*count_2), len(endtimes)
				print  (reps*count_2) < len(endtimes)-1
				if str(count_2) not in ignore_list and (reps*count_2) < (len(endtimes)-1):
					mean_epi_length.append(np.mean(endtimes[(reps*count_2):(reps*count_2)+(reps-1)]))
	else:
		mean_epi_length.append(np.mean(endtimes[0:reps-1]))
                for i in range(len(endtimes)): #0 to 6399 inclusive
                        if i % reps == 0:
                                count_2 += 1
                                if (reps*count_2) < (len(endtimes)-1):
                                        mean_epi_length.append(np.mean(endtimes[(reps*count_2):(reps*count_2)+(reps-1)]))

        return mean_epi_length

#Purpose: find duplicate parameter settings, if they exist
#http://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

#Purpose: ingoring duplicates, find the mean lengths of the epidemics
def epi_length(reps, pop):

	paras_and_means = []	
	#don't need this for postprocessing the powerlaw epidemics
	#read in the mean epidemic length and the associated parameters from powerlaw
	#with open('Jan14_pl_paras_and_means.csv', 'r') as tsv: #with open('parameters_and_means.txt', 'r') as tsv:
	#	reader1 = csv.reader(tsv, delimiter = "\t")
	#	counter = 0
	#	for row in reader1:
	#		paras_and_means.append(row)
	#		paras_and_means[counter][0] = paras_and_means[counter][0].split(',')
#
#			counter += 1
#	print 'powerlaw epidemics' #a list of lists

	#read_pl() #already done
	#read_pl_indivs(pop)

	#Calculate the mean epidemic length for the powerlaw epidemics

	count_lines = 0
	parameter_sets = [] #holds every reps^th line (every restart of an epidemic with new paras) - may contain repeats
	lines_seen = set() # holds lines already seen - no repeats
	append_first = 0

	for line in open('parameters_powerlaw_Jan14.txt', "r"):
		count_lines += 1
		if count_lines % reps ==0:
			parameter_sets.append(line)
		if line not in lines_seen: # not a duplicate
			lines_seen.add(line)

	print 'parameter_sets', len(parameter_sets)
	print 'unique lines', len(lines_seen)

	#Check if the number of parameter sets is the same as the number of unique parameter sets
	if len(lines_seen) == len(parameter_sets):
		#all is okay
		mean_epi_lengths = find_means("Jan14_powerlaw_epidemics.txt", reps)
		pl_paras_means = pl_paras_and_means(tuple(parameter_sets), mean_epi_lengths)
	else:
		print 'find duplicates'
		duplicates_ = []
		source = parameter_sets
		dups_in_source = partial(list_duplicates_of, source)
		for c in lines_seen:
			d1 = dups_in_source(c)
			dups = np.asarray(d1[1:len(d1)], dtype=np.int32)
			if len(d1) > 1:
				print 'duplicate', c, d1
				duplicates_.append(d1[1:len(d1)])
		ignore_these = [str(item) for sublist in duplicates_ for item in sublist]
		ignore_ind = [int(i) for i in ignore_these]
		total_ind = set(range(len(parameter_sets)))
		keep_ind = total_ind - set(ignore_ind)
		keep_para_list = itemgetter(*keep_ind)(parameter_sets)
		mean_epi_lengths = find_means("Jan14_powerlaw_epidemics.txt", reps, ignore_list = ignore_these)
		pl_paras_means = pl_paras_and_means(keep_para_list, mean_epi_lengths)
	print 'calculated mean epidemic lengths', mean_epi_lengths
	return (mean_epi_lengths, paras_and_means, pl_paras_means)


def pl_paras(pl_paras_means):
	#Open the parameter file for powerlaw
	para_list = []
	times = []
	with open('parameters_powerlaw_Jan14.txt', 'r') as f:
		for line in f:
			para_list.append(line.split('\n'))
	#Open the epidemic file for powerlaw
        with open('Jan14_powerlaw_epidemics.txt', 'r') as g:
                for line in g:
                        times.append(line.split(' ')[0])

        with open('Jan14_powerlaw_epidemics.txt', 'r') as g: #yes, I open it twice. Yes, I know this is bad.
                epidata = np.loadtxt(g)

        with open('Jan14_powerlaw_epidemics_individual.txt', 'r') as file1:
                indivdata = np.loadtxt(file1)

        print 'finding the ones'

        findones = find1(times) #there should be 6400 ones. These are the indices of the ones
        endtimes_indices = [findones[i]-1 for i in range(len(findones))] #now I can access the start and end of each epidemic
        endtimes_indices.pop(0) #had the empty list?
        endtimes_indices.append(epidata.shape[0]) #the last element will be the end time
	
	kept_epidemic = []
	for i in range(len(pl_paras_means)):
		pl_paras_means[i].pop(-1) #remove last element which is mean
		set1 = set([float(j) for j in pl_paras_means[i]])
		for j in range(len(para_list)):
			p1 = para_list[j][0].split(' ')
			set2 = set([float(k) for k in p1])
			if set1 == set2:
				kept_epidemic.append(j)	

#        kept_epidemic = []
#        for i in range(len(paras_kept)):
#                paras_kept[i].pop(-1)
#                set1 = set([float(j) for j in paras_kept[i]])
#                print 'set1', set1
#                for j in range(len(para_list)):
#                        p1 = para_list[j][0].split(' ')
#                        set2 = set([float(k) for k in p1])
#                        print 'set2', set2
#                        if set1 == set2:
#                                #append the index of the success to a list
#                                kept_epidemic.append(j)

        ##index the lists by what we want to keep
        findones = [findones[i] for i in kept_epidemic]
        endtimes_indices = [endtimes_indices[i] for i in kept_epidemic]

	epidemics_to_keep = []

        for k, l in zip(findones, endtimes_indices):
                epidemics_to_keep.append(epidata[k:(l+1),:]) #was l...
	
        #index the individual times to infection by what we want to keep
        counter = 0
        slice_keeper = np.array([[0., 0.], [0., 0.]])
        for i in kept_epidemic:
                for j in range((i*pop), ((i+1)*pop)):
                        counter += 1
                        print counter
                        slice_keeper = np.concatenate([slice_keeper, [indivdata[j]]]) #([slice_keeper, indivdata[(i*300):((i*300)+199)]])

        slice_keeper = slice_keeper[2:]

        print 'finished slicer'
        #indiv_to_keep = []

        #for i in kept_epidemic:
        #       indiv_to_keep.append(indivdata[i,:])
                #indiv_to_keep.append(indivdata[i,:])

        etk_times = []
        etk_infindivs = []
        etk_indiv = []
        etk_indivtime = []
        for i in range(len(epidemics_to_keep)):
                for j in range(len(epidemics_to_keep[i])):
                        etk_times.append(epidemics_to_keep[i][j][0])
                        etk_infindivs.append(epidemics_to_keep[i][j][1])

        print 'finished etk'

        if save_file:
                with open('Jan14_powerlaw_postprocessed.txt', 'w') as f:
                        np.savetxt(f, np.c_[etk_times, etk_infindivs], fmt=['%i', '%i'])

                with open('Jan14_powerlaw_individual_postprocessed.txt', 'w') as f:
                        np.savetxt(f, slice_keeper, fmt = ['%i', '%i'])			
		

#don't call this for powerlaw postprocessing
def closest_paras(paras_and_means, pl_paras_means):
	pl_mean = []
	keep_paras = []
	#print paras_and_means[0:10]
	#paras_and_means.pop(-1) #for some reason, had the empty list as the last element
	index_list = []
	already_used = []
	for i in range(len(paras_and_means)):
	     pl_mean = []
	     index_list = []
	     test_set = set([float(paras_and_means[i][0][0]), float(paras_and_means[i][0][2]), float(paras_and_means[i][0][3])]) 
	     #test_set = set([float(paras_and_means[i][0]), float(paras_and_means[i][2]), float(paras_and_means[i][3])])
	     for j in range(len(pl_paras_means)):
		  if test_set < set(pl_paras_means[j]) and j not in already_used: #was < for other types of epidemic
		       pl_mean.append(pl_paras_means[j][-1])
		       index_list.append(j)
		  else:
		       continue
	     ar1 = (np.array(float(paras_and_means[i][0][-1])) - pl_mean)**2 #was float(paras_and_means[i][-1]
	     ind1 = np.where(ar1 == ar1.min())
             get_item = index_list[ind1[0][0]]
	     already_used.append(get_item)
	     keep_paras.append(pl_paras_means[get_item])


	with open('pl_paras_listoflists.csv', 'w') as f:
		writer = csv.writer(f, delimiter = '\t')
		writer.writerows(keep_paras)

	return keep_paras


def get_epidemics_assoc(reps, save_file = False):
	#Purpose: keep the epidemics specifically associated with the parameter values we want
	#parameters_powerlaw_Jan14.txt has the parameter settings for each epidemic
	#Jan14_powerlaw_epidemics.txt has the epidemics (400 per parameter setting)

	para_list = []	
	paras_kept = []
	times = []
	epidemics_to_keep = []
	with open('parameters_powerlaw_Jan14.txt', 'r') as f:
		for line in f:
			para_list.append(line.split('\n'))#para_list is a list of the parameters assoc with each epidemic

	with open('Jan14_powerlaw_epidemics.txt', 'r') as g:
		for line in g:
			times.append(line.split(' ')[0])

        with open('Jan14_powerlaw_epidemics.txt', 'r') as g: #yes, I open it twice. Yes, I know this is bad.
		epidata = np.loadtxt(g)

	with open('pl_paras_listoflists.csv', 'r') as h:
		creader = csv.reader(h)
		for line in h:
			paras_kept.append(line.split('\t')) #paras_kept are the aprameters assoc with successful epidemics

	with open('Jan14_powerlaw_epidemics_individual.txt', 'r') as file1:
		indivdata = np.loadtxt(file1)

	print 'opened all files'
 
	for i in range(len(paras_kept)):
		paras_kept[i][-1] = paras_kept[i][-1].rstrip()

	#print type(indivdata) #a numpy.ndarray
	#print type(indivdata[0]) # numpy.ndarray
	#print indivdata.shape #(5,760,000 , 2)

	#sys.exit()

	#keep = [1, 3]
	#keepdata = []
	#slice_keeper = np.array([[0., 0.], [0., 0.]])
	#for i in keep:
	#	for j in range((i*pop), ((i+1)*pop)):
	#		slice_keeper = np.concatenate([slice_keeper, [indivdata[j]]]) #([slice_keeper, indivdata[(i*300):((i*300)+199)]])

	#slice_keeper = slice_keeper[2:]
	#print slice_keeper[0:10]

	print 'finding the ones'

	findones = find1(times) #there should be 6400 ones. These are the indices of the ones
	endtimes_indices = [findones[i]-1 for i in range(len(findones))] #now I can access the start and end of each epidemic
        endtimes_indices.pop(0) #had the empty list?
        endtimes_indices.append(epidata.shape[0]) #the last element will be the end time


	epidemics_to_keep = []

	for j in range(len(para_list)):
		para_list[j].pop(-1)

	kept_epidemic = []
	for i in range(len(paras_kept)):
		paras_kept[i].pop(-1)
		set1 = set([float(j) for j in paras_kept[i]])
		print 'set1', set1
		for j in range(len(para_list)):
			p1 = para_list[j][0].split(' ')
			set2 = set([float(k) for k in p1])
			print 'set2', set2
			if set1 == set2:
				#append the index of the success to a list
				kept_epidemic.append(j)

	print 'kept epidemics', kept_epidemic

	##index the lists by what we want to keep
        findones = [findones[i] for i in kept_epidemic]
        endtimes_indices = [endtimes_indices[i] for i in kept_epidemic]

	for k, l in zip(findones, endtimes_indices):
		epidemics_to_keep.append(epidata[k:l,:])

	print 'finished epidemics_to_keep'
	#index the individual times to infection by what we want to keep
	counter = 0
        slice_keeper = np.array([[0., 0.], [0., 0.]])
        for i in kept_epidemic:
                for j in range((i*pop), ((i+1)*pop)):
			counter += 1
			print counter
                        slice_keeper = np.concatenate([slice_keeper, [indivdata[j]]]) #([slice_keeper, indivdata[(i*300):((i*300)+199)]])

        slice_keeper = slice_keeper[2:]

	print 'finished slicer'
	#indiv_to_keep = []
	
	#for i in kept_epidemic:
	#	indiv_to_keep.append(indivdata[i,:])
		#indiv_to_keep.append(indivdata[i,:])
	
	etk_times = []
	etk_infindivs = []
	etk_indiv = []
	etk_indivtime = []
	for i in range(len(epidemics_to_keep)):
		for j in range(len(epidemics_to_keep[i])):
			etk_times.append(epidemics_to_keep[i][j][0])
			etk_infindivs.append(epidemics_to_keep[i][j][1])

	print 'finished etk'
		
	if save_file:
		with open('Jan14_powerlaw_postprocessed.txt', 'w') as f:
			np.savetxt(f, np.c_[etk_times, etk_infindivs], fmt=['%i', '%i'])

		with open('Jan14_powerlaw_individual_postprocessed.txt', 'w') as f:
			np.savetxt(f, slice_keeper, fmt = ['%i', '%i'])

	return epidemics_to_keep	

if __name__ == "__main__":
    reps = 400 #20 #400
    pop = 150 #300
    #r1 = read_pl_indivs(pop) #already called in epi_length below
    #r1 = read_pl()
    mean_pl_epi_lengths, paras_and_means, pl_para_means = epi_length(reps, pop)
    #c1 = closest_paras(paras_and_means, pl_para_means)
    ##print c1
    #print np.asarray(c1)
    #g1 = get_epidemics_assoc(reps, save_file=True)
    p2 = pl_paras(pl_para_means)
