"""
COMS W4701 Artificial Intelligence - Programming Homework 1

In this assignment you will implement and compare different search strategies
for solving the n-Puzzle, which is a generalization of the 8 and 15 puzzle to
squares of arbitrary size (we will only test it with 8-puzzles for now). 
See Courseworks for detailed instructions.

@author: Halleloya
"""

import time

def state_to_string(state):
    row_strings = [" ".join([str(cell) for cell in row]) for row in state]
    return "\n".join(row_strings)


def swap_cells(state, i1, j1, i2, j2):
    """
    Returns a new state with the cells (i1,j1) and (i2,j2) swapped. 
    """
    value1 = state[i1][j1]
    value2 = state[i2][j2]
    
    new_state = []
    for row in range(len(state)): 
        new_row = []
        for column in range(len(state[row])): 
            if row == i1 and column == j1: 
                new_row.append(value2)
            elif row == i2 and column == j2:
                new_row.append(value1)
            else: 
                new_row.append(state[row][column])
        new_state.append(tuple(new_row))
    return tuple(new_state)


def get_successors(state):
    """
    This function returns a list of possible successor states resulting
    from applicable actions. 
    The result should be a list containing (Action, state) tuples. 
    For example [("Up", ((1, 4, 2),(0, 5, 8),(3, 6, 7))), 
                 ("Left",((4, 0, 2),(1, 5, 8),(3, 6, 7)))] 
    """ 
  
    child_states = []
    
    size = len(state)
    i = 0
    j = 0
    for i in range (size):
        if 0 in state[i]:
            for j in range (size):
                if state[i][j] == 0:
                    break          
            break

    if j != size-1:
        child_states.append (("Left", swap_cells(state, i, j, i, j+1)))
    if j != 0:
        child_states.append (("Right", swap_cells(state, i, j, i, j-1)))
    if i != size-1:
        child_states.append (("Up", swap_cells(state, i, j, i+1, j)))
    if i != 0:
        child_states.append (("Down", swap_cells(state, i, j, i-1, j)))
    
    return child_states

            
def goal_test(state):
    """
    Returns True if the state is a goal state, False otherwise. 
    """      
    size = len(state)
    for i in range (size):
        for j in range (size):
            if state[i][j] != i*size + j:
                return False  
    return True

   
def bfs(state):
    """
    Breadth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a queue in BFS)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    states_expanded = 0  #Total states expanded
    max_fringe = 0  #max fringe size

    fringe = []
    closed = set()
    parents = {}
    #YOUR CODE HERE
    solution = []
    #mapping = {state:('root', 'root')}

    fringe.append(state)
    #closed.append(state)

    current_exp = state
    has_solution = False
    parents[state] = ('root', 'root')
    while len(fringe):
        current_exp = fringe.pop(0)
        states_expanded += 1
        if goal_test(current_exp):
            has_solution = True
            break
        elif current_exp in closed:
            continue
        else:
            closed.add(current_exp)
            successors = get_successors(current_exp)
            for i in range(len(successors)):
                if successors[i][1] not in parents.keys():
                    fringe.append(successors[i][1])
                    parents[successors[i][1]] = (current_exp, successors[i][0])
            if max_fringe < len(fringe):
                max_fringe = len(fringe)

    if has_solution:
        while parents[current_exp][0] != 'root' :
            solution.append(parents[current_exp][1])
            current_exp = parents[current_exp][0]
        solution.reverse()
        return solution, states_expanded, max_fringe   
    else:
        return None, states_expanded, max_fringe # No solution found
                               
     
def dfs(state):
    """
    Depth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a stack in DFS)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    states_expanded = 0
    max_fringe = 0

    fringe = []
    closed = set()
    parents = {}
    #YOUR CODE HERE
    solution = []

    fringe.append(state)

    current_exp = state
    has_solution = False
    parents[state] = ('root', 'root')
    while len(fringe):
        current_exp = fringe.pop()
        states_expanded += 1
        if goal_test(current_exp):
            has_solution = True
            break
        elif current_exp in closed:
            continue
        else:
            closed.add(current_exp)
            successors = get_successors(current_exp)
            for i in range(len(successors)):
                if successors[i][1] not in parents.keys():
                    fringe.append(successors[i][1])
                    parents[successors[i][1]] = (current_exp, successors[i][0])
            if max_fringe < len(fringe):
                max_fringe = len(fringe)

    if has_solution:
        while parents[current_exp][0] != 'root' :
            solution.append(parents[current_exp][1])
            current_exp = parents[current_exp][0]
        solution.reverse()
        return solution, states_expanded, max_fringe   
    else:
        return None, states_expanded, max_fringe # No solution found


def misplaced_heuristic(state):
    """
    Returns the number of misplaced tiles.
    """
    msp_h = 0
    size = len(state)
    for i in range (size):
        for j in range (size):
            if state[i][j] == 0:
                continue
            elif state[i][j] != i*size + j:
                msp_h += 1
    return msp_h


def manhattan_heuristic(state):
    """
    For each misplaced tile, compute the Manhattan distance between the current
    position and the goal position. Then return the sum of all distances.
    """
    man_h = 0
    size = len(state)
    for i in range (size):
        for j in range (size):
            if state[i][j] == 0:
                continue
            else:
                man_h = man_h + abs(i - int(state[i][j]/3)) + abs(j - (state[i][j])%3)
    return man_h


def best_first(state, heuristic):
    """
    Best first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a priority queue in greedy search)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    # You may want to use these functions to maintain a priority queue
    from heapq import heappush
    from heapq import heappop

    states_expanded = 0
    max_fringe = 0

    fringe = []
    closed = set()
    parents = {}

    #YOUR CODE HERE
    solution = []

    #fringe.append((heuristic(state),state))
    heappush(fringe, (heuristic(state),state))

    current_exp = state
    has_solution = False
    parents[state] = ('root', 'root')
    while len(fringe):
        current_exp = heappop(fringe)[1]
        states_expanded += 1
        if goal_test(current_exp):
            has_solution = True
            break
        elif current_exp in closed:
            continue
        else:
            closed.add(current_exp)
            successors = get_successors(current_exp)
            for i in range(len(successors)):
                if successors[i][1] not in parents.keys():
                    heappush(fringe, (heuristic(successors[i][1]),successors[i][1]))
                    #fringe.append(successors[i][1])
                    parents[successors[i][1]] = (current_exp, successors[i][0])
            if max_fringe < len(fringe):
                max_fringe = len(fringe)

    if has_solution:
        while parents[current_exp][0] != 'root' :
            solution.append(parents[current_exp][1])
            current_exp = parents[current_exp][0]
        solution.reverse()
        return solution, states_expanded, max_fringe   
    else:
        return None, states_expanded, max_fringe # No solution found


def astar(state, heuristic):
    """
    A-star search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a priority queue in greedy search)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    # You may want to use these functions to maintain a priority queue
    from heapq import heappush
    from heapq import heappop

    states_expanded = 0
    max_fringe = 0

    fringe = []
    closed = set()
    parents = {}
    costs = {}

    #YOUR CODE HERE
    solution = []

    #fringe.append((heuristic(state),state))
    heappush(fringe, (heuristic(state),state))

    current_exp = state
    has_solution = False
    parents[state] = ('root', 'root')
    costs[state] = 0
    while len(fringe):
        current_exp = heappop(fringe)[1]
        states_expanded += 1
        if goal_test(current_exp):
            has_solution = True
            break
        elif current_exp in closed:
            continue
        else:
            closed.add(current_exp)
            successors = get_successors(current_exp)
            for i in range(len(successors)):
                if successors[i][1] not in parents.keys():
                    costs[successors[i][1]] = costs[current_exp] + 1
                    heappush(fringe, (heuristic(successors[i][1]) + costs[successors[i][1]],successors[i][1]))
                    
                    #fringe.append(successors[i][1])
                    parents[successors[i][1]] = (current_exp, successors[i][0])
            if max_fringe < len(fringe):
                max_fringe = len(fringe)

    if has_solution:
        while parents[current_exp][0] != 'root' :
            solution.append(parents[current_exp][1])
            current_exp = parents[current_exp][0]
        solution.reverse()
        return solution, states_expanded, max_fringe   
    else:
        return None, states_expanded, max_fringe # No solution found

def print_result(solution, states_expanded, max_fringe):
    """
    Helper function to format test output. 
    """
    if solution is None: 
        print("No solution found.")
    else: 
        print("Solution has {} actions.".format(len(solution)))
    print("Total states expanded: {}.".format(states_expanded))
    print("Max fringe size: {}.".format(max_fringe))



if __name__ == "__main__":

    #Easy test case
    
    test_state = ((1, 4, 2),
                  (0, 5, 8), 
                  (3, 6, 7)) 
    '''
    #More difficult test case
    
    test_state = ((7, 2, 4),
                  (5, 0, 6), 
                  (8, 3, 1))'''
    

    print(state_to_string(test_state))
    print()

    print("====BFS====")
    start = time.time()
    solution, states_expanded, max_fringe = bfs(test_state) #
    end = time.time() 
    print_result(solution, states_expanded, max_fringe)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end-start))

    print() 
    print("====DFS====") 
    start = time.time()
    solution, states_expanded, max_fringe = dfs(test_state)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    print("Total time: {0:.3f}s".format(end-start))

    print() 
    print("====Greedy Best-First (Misplaced Tiles Heuristic)====") 
    start = time.time()
    solution, states_expanded, max_fringe = best_first(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    print("Total time: {0:.3f}s".format(end-start))

    
    print() 
    print("====A* (Misplaced Tiles Heuristic)====") 
    start = time.time()
    solution, states_expanded, max_fringe = astar(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    print("Total time: {0:.3f}s".format(end-start))

    print() 
    print("====A* (Total Manhattan Distance Heuristic)====") 
    start = time.time()
    solution, states_expanded, max_fringe = astar(test_state, manhattan_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    print("Total time: {0:.3f}s".format(end-start))

