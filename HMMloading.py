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
    d['pi'] = [float(i) for i in d['pi']]
    d['transitions'] = [float(i) for i in d['transitions']]
    d['emissions'] = [float(i) for i in d['emissions']]
    # Inputing the data into the HMMObject class.
    return HMMObject(d)

# Loading the sequence data in a fasta format, with sequence and latent states separated by #.
def loadseq(filename):
    rawdata = open(filename, "r").read().split(">")
    splitdata = [i.split("#") for i in rawdata[1:]]
    splitdata = [i[0].split("\n",1)+[i[1]] for i in splitdata]
    return {i[0]:(i[1].strip(), i[2].strip()) for i in splitdata}

# Loading the sequence data in a fasta format, with sequence and headers.
def readFasta(filename):
    raw = open(filename, "r").read().split(">")
    split = [i.split("\n") for i in raw[1:]]
    return {i[0]:i[1] for i in split}

# Loading the sequence data.
sequences = loadseq("sequences.txt")

# Calculating the log likelihood of the joint probability
def loglikelihood(seqpair, HMM):

    from math import log

    # Calculating for the initial pi and the initial emission
    result = log(HMM.emi[HMM.states[seqpair[1][0]],HMM.obs[seqpair[0][0]]])
    result += log(HMM.pi[HMM.states[seqpair[1][0]]])

    # Storing the previous state.
    prevstate = seqpair[1][0]

    # Iterating over all of the remaining observations and latent states
    for i in zip(seqpair[0], seqpair[1])[1:]:
        # Transitions
        result += log(HMM.trans[HMM.states[prevstate],HMM.states[i[1]]])
        # Emissions
        result += log(HMM.emi[HMM.states[i[1]],HMM.obs[i[0]]])
        prevstate = i[1]

    return result

# Loading the hidden markov model data.
hmm = loadHMM("test.hmm")