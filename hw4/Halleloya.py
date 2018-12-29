import sys, math, random

class HMM(object):
    # HMM model parameters #
    """
    Assume |X| state values and |E| emission values.
    initial: Initial belief probability distribution, list of size |X|
    tprob: Transition probabilities, size |X| list of size |X| lists;
           tprob[i][j] returns P(j|i), where i and j are states
    eprob: Emission probabilities, size |X| list of size |E| lists;
           eprob[i][j] returns P(j|i), where i is a state and j is an emission
    """
    def __init__(self, initial, tprob, eprob):
        self.initial = initial
        self.tprob = tprob
        self.eprob = eprob

    # Normalize a probability distribution
    def normalize(self, pdist):
        s = sum(pdist)
        for i in range(0,len(pdist)):
            pdist[i] = pdist[i] / s
        return pdist


    # Propagation (elapse time)
    """
    Input: Current belief distribution in the hidden state P(X_{t-1))
    Output: Updated belief distribution in the hidden state P(X_t)
    """
    def propagate(self, belief):
        length = len (belief)
        tmpblie = []
        for j in belief:
            tmpblie.append(j)
        for i in range (length):
            tmp = 0
            for j in range (length):
                tmp = tmp + self.tprob[j][i] * tmpblie[j]
            belief[i] = tmp
        belief = self.normalize(belief)
        return belief

    # Observation (weight by evidence)
    """
    Input: Current belief distribution in the hidden state P(X_t),
           index corresponding to an observation e_t
    Output: Updated belief distribution in the hidden state P(X_t | e_t)  
    """
    def observe(self, belief, obs):
        length = len (belief)
        for i in range (length):
            belief[i] = belief[i] * self.eprob[i][obs]
            #print (i, obs, self.eprob[i][obs])
        belief = self.normalize(belief)
        return belief

    # Filtering
    """
    Input: List of t observations in the form of indices corresponding to emissions
    Output: Posterior belief distribution of the hidden state P(X_t | e_1, ..., e_t)
    """
    def filter(self, observations):
        lenobs = len (observations)
        #in = self.initial
        length = len (self.initial)
        belif = []
        for i in range (length):
            belif.append (self.initial[i])
        lenblf = len (belif)
        k = 1
        while k < lenobs+1:
            b1 = self.propagate (belif)
            #b2 = self.observe (belif, observations[k-1])
            for i in range (lenblf):
                belif[i] = b1[i] * self.eprob[i][observations[k-1]] #b2[i]
            belif = self.normalize(belif)
            k = k + 1
        #self.initial = belif
        return belif #self.initial


    # Viterbi algorithm
    """
    Input: List of t observations in the form of indices corresponding to emissions
    Output: List of most likely sequence of state indices [X_1, ..., X_t]
    """
    def viterbi(self, observations):
        seq = []
        lenobs = len (observations)
        length = len (self.initial)
        belif = []
        for i in range (length):
            belif.append (self.initial[i])
        lenblf = len (belif)

        k = 1
        dic = {}
        
        while k < lenobs+1:
            b1 = []
            for i in range (length):
                tmpvalue = 0
                tmpindex = 0
                for j in range (length):
                    if self.tprob[j][i] * belif[j] > tmpvalue:
                        tmpvalue = self.tprob[j][i] * belif[j]
                        tmpindex = j
                tmax = tmpvalue
                dic [(k, i)] = tmpindex
                b1.append (tmax)
                '''
                if lenobs<15:
                    print ("No. ", k)
                    print ("bebebe", belif)
                    print ("b1b1b1", b1)
                    print ("dic", dic)
                '''
            belif = self.observe (b1, observations[k-1])
            k = k + 1
        '''
        if lenobs<15:
            print (dic)
        '''

        endmax = 0
        for tmpp in range (len(belif)):
            if belif[endmax] < belif[tmpp]:
                endmax = tmpp
        seq.append (endmax)
        while k > 2:
            k = k - 1
            seq.append (dic[(k, seq[-1])])
        '''
        if lenobs<15:
            print (dic)
            print ("seq is ", seq)
        '''
        seq.reverse()
        return seq


# Functions for testing
# You should not change any of these functions
def load_model(filename):
    model = {}
    input = open(filename, 'r')
    i = input.readline()
    x = i.split()
    model['states'] = x[0].split(",")
    
    input.readline()
    i = input.readline()
    x = i.split()
    y = x[0].split(",")
    model['initial'] = [float(i) for i in y]

    input.readline()
    tprob = []
    for i in range(len(model['states'])):
        t = input.readline()
        x = t.split()
        y = x[0].split(",")
        tprob.append([float(i) for i in y])
    model['tprob'] = tprob

    input.readline()
    i = input.readline()
    x = i.split()
    y = x[0].split(",")
    model['emissions'] = dict(zip(y, range(len(y))))

    input.readline()
    eprob = []
    for i in range(len(model['states'])):
        e = input.readline()
        x = e.split()
        y = x[0].split(",")
        eprob.append([float(i) for i in y])
    model['eprob'] = eprob

    return model

def load_data(filename):
    input = open(filename, 'r')
    data = []
    for i in input.readlines():
        x = i.split()
        if x == [',']:
            y = [' ', ' ']
        else:
            y = x[0].split(",")
        data.append(y)
    observations = []
    classes = []
    for c, o in data:
        observations.append(o)
        classes.append(c)

    data = {'observations': observations, 'classes': classes}
    return data

def generate_model(filename, states, emissions, initial, tprob, eprob):
    f = open(filename,"w+")
    for i in range(len(states)):
        if i == len(states)-1:
            f.write(states[i]+'\n')
        else:
            f.write(states[i]+',')
    f.write('\n')

    for i in range(len(initial)):
        if i == len(initial)-1:
            f.write('%f\n'%initial[i])
        else:
            f.write('%f,'%initial[i])
    f.write('\n')

    for i in range(len(states)):
        for j in range(len(states)):
            if j == len(states)-1:
                f.write('%f\n'%tprob[i][j])
            else:
                f.write('%f,'%tprob[i][j])
    f.write('\n')

    for i in range(len(emissions)):
        if i == len(emissions)-1:
            f.write(emissions[i]+'\n')
        else:
            f.write(emissions[i]+',')
    f.write('\n')

    for i in range(len(states)):
        for j in range(len(emissions)):
            if j == len(emissions)-1:
                f.write('%f\n'%eprob[i][j])
            else:
                f.write('%f,'%eprob[i][j])
    f.close()


def accuracy(a,b):
    total = float(max(len(a),len(b)))
    c = 0
    for i in range(min(len(a),len(b))):
        if a[i] == b[i]:
            c = c + 1
    return c/total

def test_filtering(hmm, observations, index_to_state, emission_to_index):
    n_obs_short = 10
    obs_short = observations[0:n_obs_short]

    print('Short observation sequence:')
    print('   ', obs_short)
    obs_indices = [emission_to_index[o] for o in observations]
    obs_indices_short = obs_indices[0:n_obs_short]

    result_filter = hmm.filter(obs_indices_short)
    result_filter_full = hmm.filter(obs_indices)

    print('\nFiltering - distribution over most recent state given short data set:')
    for i in range(0, len(result_filter)):
        print('   ', index_to_state[i], '%1.3f' % result_filter[i])

    print('\nFiltering - distribution over most recent state given full data set:')
    for i in range(0, len(result_filter_full)):
        print('   ', index_to_state[i], '%1.3f' % result_filter_full[i])

def test_viterbi(hmm, observations, classes, index_to_state, emission_to_index):
    n_obs_short = 10
    obs_short = observations[0:n_obs_short]
    classes_short = classes[0:n_obs_short]
    obs_indices = [emission_to_index[o] for o in observations]
    obs_indices_short = obs_indices[0:n_obs_short]

    result_viterbi = hmm.viterbi(obs_indices_short)
    best_sequence = [index_to_state[i] for i in result_viterbi]
    result_viterbi_full = hmm.viterbi(obs_indices)
    best_sequence_full = [index_to_state[i] for i in result_viterbi_full]

    print('\nViterbi - predicted state sequence:\n   ', best_sequence)
    print('Viterbi - actual state sequence:\n   ', classes_short)
    print('The accuracy of your viterbi classifier on the short data set is', accuracy(classes_short, best_sequence))
    print('The accuracy of your viterbi classifier on the entire data set is', accuracy(classes, best_sequence_full))


# Train a new typo correction model on a set of training data (extra credit)
"""
Input: List of t observations in the form of string or other data type literals
Output: Dictionary of HMM quantities, including 'states', 'emissions', 'initial', 'tprob', and 'eprob' 
"""
#Accuracy of your viterbi classifier on the entire data set is 0.873328198699101
def train(observations, classes):
    #observations: a sequence of temporally ordered observations, e_t
    #classes: actual hidden states associated with each emission at time t, x_t

    #state: a list of all possible hidden states i.e. "classes" for each time step
    #emissions: a list of all possible emission values, not the specific sequence of observations that's given
    #initial: initial probability distribution over states
    states = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','_']
    emissions = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','_']
    initial = []

    lenclas = len (classes)
    lenstat = len (states)
    tprob = [[0 for i in range(lenstat)] for i in range(lenstat)]
    eprob = [[0 for i in range(lenstat)] for i in range(lenstat)]
    
    #tprob
    for i in range (lenclas-1):
        p = classes[i]
        q = classes[i+1]
        tprob [states.index(p)][states.index(q)] = tprob [states.index(p)][states.index(q)] + 1
    for i in range (lenstat):
        tmps = 0
        for j in range (lenstat):
            tmps += tprob [i][j]
        for j in range (lenstat):
            tprob [i][j] = (tprob [i][j] + 1) / (tmps + lenstat)

    #eprob
    for i in range (lenclas):
        p = classes[i]
        q = observations[i]
        eprob [states.index(p)][emissions.index(q)] = eprob [states.index(p)][emissions.index(q)] + 1
    for i in range (lenstat):
        tmps = 0
        for j in range (lenstat):
            tmps += eprob [i][j]
        for j in range (lenstat):
            eprob [i][j] = (eprob [i][j] + 1) / (tmps + lenstat)

    #initial
    for i in states:
        num = classes.count(i)
        initial.append((num + 1)/(lenclas + lenstat))
    #print ("initial", initial)
    #print ("tprob", tprob)
    #print ("eprob", eprob)
    return {'states': states, 'emissions': emissions, 'initial': initial, 'tprob': tprob, 'eprob': eprob}

if __name__ == '__main__':
    # this if clause for extra credit training only
    if len(sys.argv) == 4 and sys.argv[1] == '-t':
        input = open(sys.argv[3], 'r')
        data = []
        for i in input.readlines():
            x = i.split()
            if x == [',']:
                y = [' ', ' ']
            else:
                y = x[0].split(",")
            data.append(y)

        observations = []
        classes = []
        for c, o in data:
            observations.append(o)
            classes.append(c)

        model = train(observations, classes)
        generate_model(sys.argv[2], model['states'], model['emissions'], model['initial'], model['tprob'], model['eprob'])
        exit(0)

    # main part of the assignment
    if len(sys.argv) != 3:
        print("\nusage: ./hmm.py [model file] [data file]")
        exit(0)

    model = load_model(sys.argv[1])
    data = load_data(sys.argv[2])

    new_hmm = HMM(model['initial'], model['tprob'], model['eprob'])
    test_filtering(new_hmm, data['observations'], model['states'], model['emissions'])
    test_viterbi(new_hmm, data['observations'], data['classes'], model['states'], model['emissions'])
