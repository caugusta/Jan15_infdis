import numpy as np
import re


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

	return np.c_[times, infindivs]

def read_pl_new():

        with open( 'epidemics_powerlaw_Jan14.json', 'r' ) as f :
            data = f.read()

        count = len(data.split('}, {'))
        print 'exponential epidemics', count

        str_epi = ' '.join(data.split('{'))
        str_epi2 = str_epi.split('}')

        times = re.findall(r'"(.\w*)"', str_epi2[0])
        infindivs = re.findall(r':(.\w*),', str_epi2[0])

        lastone = str_epi2[0].rfind(':') #last occurrance of :
        new1 = infindivs
        new1.append(str_epi2[0][(lastone+2):]) ## lastone: is ': number'. Don't want the colon or the space

        times = np.asarray(times, dtype=np.int32)
        infindivs = np.asarray(infindivs, dtype=np.int32)
        newinf = infindivs

        i=1
        while i < len(str_epi2):
                times_1 = re.findall(r'"(.\w*)"', str_epi2[i])
                infindivs_1 = re.findall(r':(.\w*),', str_epi2[i])
                lastone = str_epi2[i].rfind(':') #str_epi2[0][-1]
                newinf = infindivs_1
                newinf.append(str_epi2[i][(lastone+2):])
                if len(newinf) < 2:
                        print 'list was empty'
                        i += 1
                        continue
                else:
                        times = np.concatenate((times, np.asarray(times_1, dtype=np.int32)))
                        infindivs = np.concatenate((infindivs, np.asarray(infindivs_1, dtype=np.int32)))
                        i += 1
	return np.c_[times, infindivs]

if __name__ == "__main__":
	d1 = read_pl()
	d2 = read_pl_new()

	print 'original', d1[501:1000]
	print 'new', d2[501:1000]

