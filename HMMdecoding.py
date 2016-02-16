from numpy import matrix

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
    if x <= 0:
        return float("-inf")
    return log(x)
    
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

import numpy as np

obs = sequences['KDGL_ECOLI']
M = np.zeros((3,len(obs)))
i = 'i'*len(obs)
o = 'o'*len(obs)
m = 'M'*len(obs) 
seqpair_i = (obs, i)
seqpair_o = (obs, o)
seqpair_m = (obs, m)

#print seqpair_i[1][0] # states 
#print seqpair_i[0][1] # observables
prob_i = loglikelihood_M(seqpair_i, hmm)
prob_o = loglikelihood_M(seqpair_o, hmm)
prob_m = loglikelihood_M(seqpair_m, hmm)
def filling_up(matrix, prob, position):
    for i in range(len(prob)):
        matrix[position,i] = prob[i]
    return M

M = filling_up(M, prob_i, 0)
M = filling_up(M, prob_o, 1)
M = filling_up(M, prob_m, 2)

#print M

def Viterbi(seq, hmm):
    # Initialize the omega table
    N = len(seq)
    M = np.zeros((len(hmm.states), N))
    M.fill(float("-inf"))

    # Fill in the first column.
    for v in hmm.states.values():
        M[v,0] = hmm.pi[v]+hmm.emi[v,hmm.obs[seq[0]]]

    # Fill in the remaining columns in the table.
    n = 1
    for i in seq[1:]:
        o = hmm.obs[i]
        for k in hmm.states.values():
            if hmm.emi[k,o]!=float("-inf"):
                for j in hmm.states.values():
                    if hmm.trans[k,j]!=float("-inf"):
                        M[k,n] = max([M[k,n], M[j, n-1]+hmm.emi[k,o]+hmm.trans[j,k]])
        n += 1
        
    # Backtracking:
    z = ['' for i in range(len(seq))]

    # Find the last max:
    z[N-1] = hmm.states.keys()[M[:,N-1].argmax()]
    for n in range(N-1)[::-1]:
        z[n] = hmm.states.keys()[M[:,n].argmax()]

    z = ["o" if i=="i" else i for i in z]
        
    print "".join(z)
    print loglikelihood((seq, z), hmm)

    #Backtrack.
    for n in range(N-1)[::-1]:
        temp = np.array([float("-inf") for i in range(len(hmm.states))])
        o, ns = hmm.obs[seq[n+1]], hmm.states[z[n+1]]
        for i in hmm.states.values():
            temp[i] = hmm.emi[i,o]+M[i,n]+hmm.trans[i, ns]
        z[n] = hmm.states.keys()[temp.argmax()]
        #print

    print "".join(z)
    
    return z


print loglikelihood((obs,"".join(Viterbi(obs, hmm))), hmm)

#print "".join(Viterbi(sequences["RFBP_SALTY"], hmm))

#print "".join(Viterbi(sequences["FTSH_ECOLI"], hmm))
            
def Posterior():
    pass
