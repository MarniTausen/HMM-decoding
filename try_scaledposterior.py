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

    from copy import deepcopy


hmmP = deepcopy(hmm)

eexp = np.vectorize(eexp)

hmmP.emi = eexp(hmmP.emi)

hmmP.trans = eexp(hmmP.trans)

hmmP.pi = eexp(hmmP.pi)

zobs = PosteriorScaled(sequences["FTSH_ECOLI"], hmmP)

print zobs

print loglikelihood((sequences["FTSH_ECOLI"], zobs), hmm)


