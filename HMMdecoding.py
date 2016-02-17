from numpy import matrix
import numpy as np

# Object for storing all of the HMM (Hidden Markov Model) data inside.
# Takes the output from loadHMM() and makes the data more accessible
class HMMObject(object):

    """
    Contains the following variables: states (latent states), obs (observables), 
    pi (priori probabilities), trans (transition probabilities), emi (emission probabilities)"""
    
    # Initialization of object, preparing all of the datatypes.
    # States and obs get converted into a dictionary with the letters as keys,
    # and the values are the corresponding indices.
    # Trans and emi get converted into a nested list, with 3 internal lists.
    def __init__(self, hmmdict):
        self.d = hmmdict
        self.states = {self.d['hidden'][i]:i for i in range(len(self.d['hidden']))}
        self.obs = {self.d['observables'][i]:i for i in range(len(self.d['observables']))}
        self.pi = self.d['pi']
        self.trans = matrix(self.makenested(self.d['transitions'], 3))
        self.emi = matrix(self.makenested(self.d['emissions'], 3))

    # Function splits a list into a nested list, into r parts. (r meaning rows)
    def makenested(self, x, r):
        n = len(x)/r
        result = []
        for i in range(r):
            result.append(x[i*n:n*(i+1)])
        return result

def elog(x):
    from math import log
    if x == 0:
        return float("-inf")
    return log(x)

def eexp(x):
    from math import exp
    if x == float("-inf"):
        return 0
    return exp(x)

# Loading the hidden markov model (hmm) data.
# Fist by splitting the for each of the names.
# Then collecting all of the data with the right labels in a dictionary.
# Then the necessary data conversion
def loadHMM(filename):
    import re
    # Loading the data
    rawdata = open(filename, "r").read().replace("\n", " ")
    # Splitting by name
    splitdata = re.split("hidden|observables|pi|transitions|emissions", rawdata)
    splitdata = [i.strip().split(" ") for i in splitdata[1:]]
    # Collecting the data in a dictionary
    labels = ["hidden", "observables", "pi", "transitions", "emissions"]
    d = {labels[i]:splitdata[i] for i in range(len(splitdata))}
    # Data conversion
    d['pi'] = [elog(float(i)) for i in d['pi']]
    d['transitions'] = [elog(float(i)) for i in d['transitions']]
    d['emissions'] = [elog(float(i)) for i in d['emissions']]
    # Inputing the data into the HMMObject class.
    return HMMObject(d)

# Loading the sequence data in a fasta format, with sequence and latent states separated by #.
def loadseq(filename):
    rawdata = open(filename, "r").read().split(">")
    splitdata = [i.split("#") for i in rawdata[1:]]
    splitdata = [i[0].split("\n",1)+[i[2]] for i in splitdata]
    return {i[0]:(i[1].strip(), i[2].strip()) for i in splitdata}

# Loading the sequence data in a fasta format, with sequence and headers.
def readFasta(filename):
    import re
    raw = open(filename, "r").read().replace("\n", " ").strip().split(">")
    split = [re.split("\s*", i) for i in raw[1:]]
    return {i[0]:i[1] for i in split}

# Calculating the log likelihood of the joint probability
def loglikelihood(seqpair, HMM):

    from math import log

    # Calculating for the initial pi and the initial emission
    result = HMM.emi[HMM.states[seqpair[1][0]],HMM.obs[seqpair[0][0]]]
    result += HMM.pi[HMM.states[seqpair[1][0]]]

    # Storing the previous state.
    prevstate = seqpair[1][0]

    # Iterating over all of the remaining observations and latent states
    for i in zip(seqpair[0], seqpair[1])[1:]:
        # Transitions
        result += HMM.trans[HMM.states[prevstate],HMM.states[i[1]]]
        # Emissions
        result += HMM.emi[HMM.states[i[1]],HMM.obs[i[0]]]
        prevstate = i[1]

    return result

# Loading the hidden markov model data.
hmm = loadHMM("hmm-tm.txt")

# Loading the sequence data.
sequences = readFasta("sequences-project2.txt")


def loglikelihood_M(seqpair, HMM):
    result = []
    # Calculating for the initial pi and the initial emission
    result.append(HMM.emi[HMM.states[seqpair[1][0]],HMM.obs[seqpair[0][0]]])
    result[0] += HMM.pi[HMM.states[seqpair[1][0]]]

    # Storing the previous state.
    prevstate = seqpair[1][0]

    # Iterating over all of the remaining observations and latent states
    for i in zip(seqpair[0], seqpair[1])[1:]:
        # Transitions
        trans = (HMM.trans[HMM.states[prevstate],HMM.states[i[1]]])
        # Emissions
        emi = HMM.emi[HMM.states[i[1]],HMM.obs[i[0]]]
        result.append(trans+emi)
        prevstate = i[1]

    return result

#print hmm.d

#print sequences
# just taking one of the sequences to test

def Viterbi(seq, hmm):
    # Initialize the omega table
    N = len(seq)
    M = np.zeros((len(hmm.states), N))
    M.fill(float("-inf"))

    # Fill in the first column.
    for k in hmm.states.values():
        M[k,0] = hmm.pi[k]+hmm.emi[k,hmm.obs[seq[0]]]

    # Fill in the remaining columns in the table.
    n = 1
    for i in seq[1:]:
        o = hmm.obs[i]
        for k in hmm.states.values():
            if hmm.emi[k,o]!=float("-inf"):
                for j in hmm.states.values():
                    if hmm.trans[j,k]!=float("-inf"):
                        M[k,n] = max([M[k,n], M[j, n-1]+hmm.emi[k,o]+hmm.trans[j,k]])
        n += 1
        
    # Backtracking:
    z = ['' for i in range(len(seq))]

    # Find the last max:
    z[N-1] = hmm.states.keys()[M[:,N-1].argmax()]
    for n in range(N-1)[::-1]:
        z[n] = hmm.states.keys()[M[:,n].argmax()]

    z = ["o" if i=="i" else i for i in z]


    #Backtrack.
    for n in range(N-1)[::-1]:
        temp = np.array([float("-inf") for i in range(len(hmm.states))])
        o, ns = hmm.obs[seq[n+1]], hmm.states[z[n+1]]
        for i in hmm.states.values():
            temp[i] = hmm.emi[i,o]+M[i,n]+hmm.trans[i, ns]
        z[n] = hmm.states.keys()[temp.argmax()]
        #print

    return "".join(z)

#just printing the values
#for key in sorted(sequences):
#    temp_viterbi = Viterbi(sequences[key], hmm)
#    print'>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (key, sequences[key], temp_viterbi, loglikelihood((sequences[key], temp_viterbi), hmm))

#saving into a file:
output = str()
for key in sorted(sequences):
    temp_viterbi = Viterbi(sequences[key], hmm)
    output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (key, sequences[key], temp_viterbi, loglikelihood((sequences[key], temp_viterbi), hmm))
file = open('output_viterbi.txt', "w")
file.write(output)
file.close()

#making the logsum to transform the Posterior code
def LOGSUM(x, y): #the input is already log transformed
    if x == float("-inf"):
        return y
    if y == float("-inf"):
        return x
    if x > y:
        return x + elog(1 + 2**(y - x)) 
    else:
        return y + elog(1 + 2**(x - y))


def Posterior(seq, hmm):
    # Initializing the tables
    N = len(seq)
    A = np.zeros((len(hmm.states), N))
    A.fill(float("-inf"))
    B = np.zeros((len(hmm.states), N))
    B.fill(float("-inf"))

    # Filling up the alpha table:
    for k in hmm.states.values():
        A[k,0] = hmm.pi[k]+hmm.emi[k,hmm.obs[seq[0]]]
    
    for n in range(1,N):
        o = hmm.obs[seq[n]]
        for k in hmm.states.values():
            logsum = float("-inf")
            if hmm.emi[k,o]!=float("-inf"):
                for j in hmm.states.values():
                    if hmm.trans[j,k]!=float("-inf"):
                        logsum = LOGSUM(logsum, A[j, n-1]+hmm.trans[j,k])
            if logsum!=float("-inf"):
                logsum += hmm.emi[k,o]
            A[k,n] = logsum
    
    # Filling up the beta table:
    B[:,N-1] = elog(1)
    for n in range(0,N-1)[::-1]:
        o = hmm.obs[seq[n]]
        for k in hmm.states.values():
            logsum = float("-inf")
            if hmm.emi[k,o]!=float("-inf"):
                for j in hmm.states.values():
                    if hmm.trans[k,j]!=float("-inf"):
                        logsum = LOGSUM(logsum, B[j, n+1]+hmm.trans[k,j])
            if logsum!=float("-inf"):
                logsum += hmm.emi[k,o]
            B[k,n] = logsum

    # Posterior decoding:
    M = A+B

    z = ['' for i in range(len(seq))]
    for n in range(N):
        z[n] = hmm.states.keys()[M[:,n].argmax()]
    
    return "".join(z)

zobs = Posterior(sequences["FTSH_ECOLI"], hmm)

print zobs

#print loglikelihood((sequences["FTSH_ECOLI"], zobs), hmm)


# output = str()
# for key in sorted(sequences):
#     temp_post = Posterior(sequences[key], hmm)
#     output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (key, sequences[key], temp_post, loglikelihood((sequences[key], temp_post), hmm))
# file = open('output_posterior.txt', "w")
# file.write(output)
# file.close()

from copy import deepcopy


hmmP = deepcopy(hmm)

for x in range(hmmP.emi.shape[0]):
    for y in range(hmmP.emi.shape[1]):
        hmmP.emi[x,y] = eexp(hmmP.emi[x,y])

for x in range(hmmP.trans.shape[0]):
    for y in range(hmmP.trans.shape[1]):
        hmmP.trans[x,y] = eexp(hmmP.trans[x,y])

for x in range(len(hmmP.pi)):
    hmmP.emi[x,y] = eexp(hmmP.emi[x,y])

def PosteriorScaled(seq, hmm):
    # Initializing the tables
    N = len(seq)
    A = np.zeros((len(hmm.states), N))
    A.fill(0)
    B = np.zeros((len(hmm.states), N))
    B.fill(0)

    c = []
    
    # Filling up the alpha table:
    for k in hmm.states.values():
        A[k,0] = hmm.pi[k]*hmm.emi[k,hmm.obs[seq[0]]]

    c.append(A[:,0].sum())
    A[:,0] = A[:,0]/c[0]

    
    for n in range(1,N):
        o = hmm.obs[seq[n]]
        for k in hmm.states.values():
            thesum = 0
            if hmm.emi[k,o]!=0:
                for j in hmm.states.values():
                    if hmm.trans[j,k]!=0:
                        thesum = thesum+A[j, n-1]*hmm.trans[j,k]
                A[k,n] = thesum*hmm.emi[k,o]
        c.append(A[:,n].sum())
        A[:,n] = A[:,n]/c[n]

    print A[:,0:6]
        
    # Filling up the beta table:
    B[:,N-1] = 1
    c[N-1] = B[:,N-1].sum()
    B[:,N-1] = B[:,N-1]/c[N-1]
    for n in range(0,N-1)[::-1]:
        o = hmm.obs[seq[n]]
        for k in hmm.states.values():
            thesum = 0
            if hmm.emi[k,o]!=0:
                for j in hmm.states.values():
                    if hmm.trans[k,j]!=0:
                        thesum = thesum+B[j, n+1]*hmm.trans[k,j]
                B[k,n] = thesum*hmm.emi[k,o]
        c[n] = B[:,n].sum()
        B[:,n] = B[:,n]/c[n]

    print B[:,N-6:N]
    # Posterior decoding:
    M = A*B

    z = ['' for i in range(len(seq))]
    #for n in range(N):
    #    z[n] = hmm.states.keys()[M[:,n].argmax()]

    for n in range(N):
        cmax, kmax = 0, ''
        maxed = False
        for k, v in hmm.states.items():
            if M[v,n]==cmax:
                maxed = True
            if M[v,n]>cmax:
                cmax = M[v,n]
                kmax = k
        if maxed==True:
            print M[:, n]
        z[n] = kmax
            
    
    return "".join(z)

zobs = PosteriorScaled(sequences["FTSH_ECOLI"], hmmP)

print zobs

print loglikelihood((sequences["FTSH_ECOLI"], zobs), hmm)

# output = str()
# for key in sorted(sequences):
#     temp_post = PosteriorScaled(sequences[key], hmm)
#     output += '>%s \n%s \n#\n%s\n; log P(x,z) = %f\n' % (key, sequences[key], temp_post, loglikelihood((sequences[key], temp_post), hmm))
# file = open('output_posteriorScaled.txt', "w")
# file.write(output)
# file.close()


mat = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print mat

print mat[:,0].sum(), mat[:,1].sum(), mat[:,2].sum()


