import random
import math
import sys

class FrozenLake(object):

    def __init__(self, width, height, start, targets, blocked, holes):
        self.initial_state = start 
        self.width = width
        self.height = height
        self.targets = targets
        self.holes = holes
        self.blocked = blocked

        self.actions = ('n', 's', 'e', 'w')
        self.states = set()
        for x in range(width):
            for y in range(height):
                if (x,y) not in self.targets and (x,y) not in self.holes and (x,y) not in self.blocked:
                    self.states.add((x,y))

        # Parameters for the simulation
        self.gamma = 0.9
        self.success_prob = 0.8
        self.hole_reward = -5.0
        self.target_reward = 1.0
        self.living_reward = -0.1

    #### Internal functions for running policies ###

    def get_transitions(self, state, action):
        """
        Return a list of (successor, probability) pairs that
        can result from taking action from state
        """
        result = []
        x,y = state
        remain_p = 0.0

        if action=="n":
            success = (x,y-1)
            fail = [(x+1,y), (x-1,y)]
        elif action=="s":
            success =  (x,y+1)
            fail = [(x+1,y), (x-1,y)]
        elif action=="e":
            success = (x+1,y)
            fail= [(x,y-1), (x,y+1)]
        elif action=="w":
            success = (x-1,y)
            fail= [(x,y-1), (x,y+1)]
          
        if success[0] < 0 or success[0] > self.width-1 or \
           success[1] < 0 or success[1] > self.height-1 or \
           success in self.blocked: 
                remain_p += self.success_prob
        else: 
            result.append((success, self.success_prob))
        
        for i,j in fail:
            if i < 0 or i > self.width-1 or \
               j < 0 or j > self.height-1 or \
               (i,j) in self.blocked: 
                    remain_p += (1-self.success_prob)/2
            else: 
                result.append(((i,j), (1-self.success_prob)/2))
           
        if remain_p > 0.0: 
            result.append(((x,y), remain_p))
        return result

    def move(self, state, action):
        """
        Return the state that results from taking this action
        """
        transitions = self.get_transitions(state, action)
        new_state = random.choices([i[0] for i in transitions], weights=[i[1] for i in transitions])
        return new_state[0]

    def simple_policy_rollout(self, policy):
        """
        Return (Boolean indicating success of trial, total rewards) pair
        """
        state = self.initial_state
        rewards = 0
        while True:
            if state in self.targets:
                return (True, rewards+self.target_reward)
            if state in self.holes:
                return (False, rewards+self.hole_reward)
            state = self.move(state, policy[state])
            rewards += self.living_reward

    def QValue_to_value(self, Qvalues):
        """
        Given a dictionary of q-values corresponding to (state, action) pairs,
        return a dictionary of optimal values for each state
        """
        values = {}
        for state in self.states:
            values[state] = -float("inf")
            for action in self.actions:
                values[state] = max(values[state], Qvalues[(state, action)])
        return values


    #### Some useful functions for you to visualize and test your MDP algorithms ###

    def test_policy(self, policy, t=500):
        """
        Following the policy t times, return (Rate of success, average total rewards)
        """
        numSuccess = 0.0
        totalRewards = 0.0
        for i in range(t):
            result = self.simple_policy_rollout(policy)
            if result[0]:
                numSuccess += 1
            totalRewards += result[1]
        return (numSuccess/t, totalRewards/t)

    def get_random_policy(self):
        """
        Generate a random policy.
        """
        policy = {}
        for i in range(self.width):
            for j in range(self.height):
                policy[(i,j)] = random.choice(self.actions)
        return policy
    '''
    def gen_rand_set(width, height, size):
        """
        Generate a random set of grid spaces.
        Useful for creating randomized maps.
        """
        mySet = set([])
        while len(mySet) < size:
            mySet.add((random.randint(0, width), random.randint(0, height)))
        return mySet'''


    def print_map(self, policy=None):
        """
        Print out a map of the frozen pond, where in indicates start state,
        T indicates target states, # indicates blocked states, and O indicates holes.
        A policy may optimally be provided, which will be printed out on the map as well.
        """
        sys.stdout.write(" ")
        for i in range(2*self.width):
            sys.stdout.write("--")
        sys.stdout.write("\n")
        for j in range(self.height):
            sys.stdout.write("|")
            for i in range(self.width):
                if (i, j) in self.targets:
                    sys.stdout.write("T\t")
                elif (i, j) in self.holes:
                    sys.stdout.write("O\t")
                elif (i, j) in self.blocked:
                    sys.stdout.write("#\t")
                else:
                    if policy and (i, j) in policy:
                        a = policy[(i, j)]
                        if a == "n":
                            sys.stdout.write("^")
                        elif a == "s":
                            sys.stdout.write("v")
                        elif a == "e":
                            sys.stdout.write(">")
                        elif a == "w":
                            sys.stdout.write("<")
                        sys.stdout.write("\t")
                    elif (i, j) == self.initial_state:
                        sys.stdout.write("*\t")
                    else:
                        sys.stdout.write(".\t")
            sys.stdout.write("|")
            sys.stdout.write("\n")
        sys.stdout.write(" ")
        for i in range(2*self.width):
            sys.stdout.write("--")
        sys.stdout.write("\n")

    def print_values(self, values):
        """
        Given a dictionary {state: value}, print out the values on a grid
        """
        for j in range(self.height):
            for i in range(self.width):
                if (i, j) in self.holes:
                    value = self.hole_reward
                elif (i, j) in self.targets:
                    value = self.target_reward
                elif (i, j) in self.blocked:
                    value = 0.0
                else:
                    value = values[(i, j)]
                print("%10.2f" % value, end='')
            print()


    #### Your code starts here ###

    def value_iteration(self, threshold=0.001):
        """
        The value iteration algorithm to iteratively compute an optimal
        value function for all states.
        """
        values = dict((state, 0.0) for state in self.states)
        ### YOUR CODE HERE ###
        flag = True
        #self.print_values(values)
        #t = 0
        while flag:
                flag = False
                #t = t+1
                value_tmp = {}
                for state in self.states:
                        max_action = float('-inf') 
                        for action in self.actions:
                                trans_result = self.get_transitions(state, action)				
                                v = 0
                                for trans in trans_result:
                                        if trans[0] in self.targets:
                                                v = v + trans[1] * (self.living_reward + self.gamma * self.target_reward)
                                        elif trans[0] in self.holes:
                                                v = v + trans[1] * (self.living_reward + self.gamma * self.hole_reward)
                                        else:
                                                v = v + trans[1] * (self.living_reward + self.gamma * values[trans[0]])
                                                                
                                if v > max_action:
                                        max_action = v
                        value_tmp[state] = max_action
                        #print (state)
                        #print(value_tmp[state])
                for state in self.states:
                        if (value_tmp[state] - values[state]) > threshold:
                                flag = True
                        values[state] = value_tmp[state]
                #print ('')
                #self.print_values(values)
                #print (t)
        return values

    def extract_policy(self, values):
        """
        Given state values, return the best policy.
        """
        policy = {}
        ### YOUR CODE HERE ###
        for state in self.states:
                max_action = float('-inf')
                act = self.actions[0]
                for action in self.actions:
                        trans_result = self.get_transitions(state, action)
                        v = 0
                        for trans in trans_result:
                                if trans[0] in self.targets:
                                        v = v + trans[1] * (self.living_reward + self.gamma * self.target_reward)
                                elif trans[0] in self.holes:
                                        v = v + trans[1] * (self.living_reward + self.gamma * self.hole_reward)
                                else:
                                        v = v + trans[1] * (self.living_reward + self.gamma * values[trans[0]])
                                                                        
                        if v > max_action:
                                max_action = v
                                act = action
                policy[state] = act
        return policy


    def Qlearner(self, alpha, epsilon, num_robots):
        """
        Implement Q-learning with the alpha and epsilon parameters provided.
        Runs number of episodes equal to num_robots.
        """
        Qvalues = {}
        for state in self.states:
            for action in self.actions:
                Qvalues[(state, action)] = 0

        ### YOUR CODE HERE ###
        for i in range (num_robots):
            #This part of code that ''' included is for part 4
            '''
            epsilon = epsilon * (1 - i/num_robots)
            if i > num_robots/3:
                alpha = alpha/3
            elif i > 2*num_robots/3:
                alpha = alpha/6
            '''
            meethole = True
            while meethole:
                #print('robot', i, 'start')
                meethole = False
                state = self.initial_state
                while True:
                    maxQ = float('-inf')
                    act = self.actions[0]
                    for action in self.actions:
                        if Qvalues[(state, action)] > maxQ:
                            maxQ = Qvalues[(state, action)]
                            act = action
                    if random.random() < epsilon:
                        act = random.choice(self.actions)
                    tmpstate = self.move(state, act)
                    if tmpstate in self.holes:
                        Qvalues[(state, act)] = (1-alpha)*Qvalues[(state, act)] + alpha*(self.living_reward + self.gamma * self.hole_reward)
                        meethole = True
                        break
                    elif tmpstate in self.targets:
                        Qvalues[(state, act)] = (1-alpha)*Qvalues[(state, act)] + alpha*(self.living_reward + self.gamma * self.target_reward)
                        break
                    else:
                        tmpmaxQ = float('-inf')
                        for action in self.actions:
                            if Qvalues[(tmpstate, action)] > tmpmaxQ:
                                #print((tmpstate, action))
                                tmpmaxQ = Qvalues[(tmpstate, action)]
                        Qvalues[(state, act)] = (1-alpha)*Qvalues[(state, act)] + alpha*(self.living_reward + self.gamma * tmpmaxQ)
                    state = tmpstate  
            
        return Qvalues
    
def gen_rand_set(width, height, size):
        """
        Generate a random set of grid spaces.
        Useful for creating randomized maps.
        """
        mySet = set([])
        while len(mySet) < size:
            mySet.add((random.randint(0, width), random.randint(0, height)))
        return mySet

if __name__ == "__main__":
   
    # Create a lake simulation
    width = 8
    height = 8
    start = (0,0)
    targets = set([(3,4)])
    blocked = set([(3,3), (2,3), (2,4)])
    holes = set([(4, 0), (4, 1), (3, 0), (3, 1), (6, 4), (6, 5), (0, 7), (0, 6), (1, 7)])
    lake = FrozenLake(width, height, start, targets, blocked, holes)

    rand_policy = lake.get_random_policy()
    lake.print_map()
    lake.print_map(rand_policy)
    print(lake.test_policy(rand_policy))

    opt_values = lake.value_iteration()
    lake.print_values(opt_values)
    opt_policy = lake.extract_policy(opt_values)
    lake.print_map(opt_policy)
    print(lake.test_policy(opt_policy))
    #
    Qvalues = lake.Qlearner(alpha=0.5, epsilon=0.5, num_robots=100)
    #when num_robots=80, there are 4 over 50 times that total reward greater than -1
    #when num_robots=90, there are 3 over 50 times that total reward slightly greater than -1
    #when num_robots=90, there are 2 over 50 times that total reward slightly greater than -1
    learned_values = lake.QValue_to_value(Qvalues)
    learned_policy = lake.extract_policy(learned_values)
    lake.print_map(learned_policy)
    print(lake.test_policy(learned_policy))

    #Part 4
    
    print("Below is part 4, random generate and tune")
    width = 9
    height = 9
    
    blocked = gen_rand_set(width-1, height-1, 3)
    holes = gen_rand_set(width-1, height-1, 3)
    targets = gen_rand_set(width-1, height-1, 2)
    start = (0,0)
    lake = FrozenLake(width, height, start, targets, blocked, holes)
    while start in blocked or start in holes or start in targets:
        blocked = gen_rand_set(width, height, 3)
        holes = gen_rand_set(width, height, 3)
        targets = gen_rand_set(width, height, 2)
        lake = FrozenLake(width, height, start, targets, blocked, holes)
    lake.print_map()
    
    opt_values = lake.value_iteration()
    lake.print_values(opt_values)
    opt_policy = lake.extract_policy(opt_values)
    lake.print_map(opt_policy)
    print(lake.test_policy(opt_policy))
    #
    Qvalues = lake.Qlearner(alpha=0.8, epsilon=0.8, num_robots=50)
    learned_values = lake.QValue_to_value(Qvalues)
    learned_policy = lake.extract_policy(learned_values)
    lake.print_map(learned_policy)
    print(lake.test_policy(learned_policy))
    
