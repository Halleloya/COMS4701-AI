#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
COMS W4701 Artificial Intelligence - Programming Homework 2

An AI player for Othello. This is the template file that you need to  
complete and submit. 

@author: Halleloya 
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI 
from othello_shared import find_lines, get_possible_moves, get_score, play_move

history = {}

def compute_utility(board, color):
    """
    Return the utility of the given board state
    (represented as a tuple of tuples) from the perspective
    of the player "color" (1 for dark, 2 for light)
    """
    p1_count, p2_count = get_score(board)
    utility = p1_count - p2_count
    if color == 2:
        utility = 0 - utility
    return utility


############ MINIMAX ###############################

def minimax_min_node(board, color):
    if color == 1:
        neg_color = 2
    else:
        neg_color = 1
    actions = get_possible_moves(board, neg_color)
    if len(actions) == 0:
        return compute_utility (board, color)
    min_uti = minimax_max_node(play_move(board,neg_color,actions[0][0],actions[0][1]),color)
    for i in range(1,len(actions)):
        tmp = minimax_max_node(play_move(board,neg_color,actions[i][0],actions[i][1]),color)
        if tmp < min_uti:
            min_uti = tmp
            
    return min_uti


def minimax_max_node(board, color):
    actions = get_possible_moves(board, color)
    if len(actions) == 0:
        return compute_utility(board, color)
    max_uti = minimax_min_node(play_move(board,color,actions[0][0],actions[0][1]),color)
    for i in range(1,len(actions)):
        tmp = minimax_min_node(play_move(board,color,actions[i][0],actions[i][1]),color)
        if tmp > max_uti:
            max_uti = tmp

    return max_uti

    
def select_move_minimax(board, color):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  
    """
    actions = get_possible_moves(board, color)
    if not actions:
        return 0, 0
    max_uti = minimax_min_node(play_move(board,color,actions[0][0],actions[0][1]),color)
    res = actions[0]
    for i in range(1,len(actions)):
        tmp = minimax_min_node(play_move(board,color,actions[i][0],actions[i][1]),color)
        if tmp > max_uti:
            max_uti = tmp
            res = actions[i]

    return res[0], res[1]


############ ALPHA-BETA PRUNING #####################

#alphabeta_min_node(board, color, alpha, beta, level, limit)
def alphabeta_min_node(board, color, alpha, beta,level,limit):
    if level > limit:
        return compute_utility(board,color)
    if board in history:
        return history[board]
    if color == 1:
        neg_color = 2
    else:
        neg_color = 1
    actions = get_possible_moves(board, neg_color)
    if len(actions) == 0:
        return compute_utility(board,color)
    min_uti = float('inf')
    boards = []
    for action in actions:
        boards.append(play_move(board,neg_color,action[0],action[1]))
    QuickSort(boards,0,len(boards)-1,neg_color)

    for board_tmp in boards:
        tmp = alphabeta_max_node(board_tmp,color,alpha,beta,level+1,limit)
        if tmp == None:
            continue
        min_uti = min(min_uti,tmp)
        if min_uti < alpha or min_uti == alpha:
            history[board] = min_uti
            return min_uti
        beta = min(beta,min_uti)
    history[board] = min_uti
    return min_uti


#alphabeta_max_node(board, color, alpha, beta, level, limit)
def alphabeta_max_node(board, color, alpha, beta,level,limit):
    if level > limit:
        return compute_utility(board,color)  
    if board in history:
        return history[board]
    actions = get_possible_moves(board, color)
    if len(actions) == 0:
        return compute_utility(board,color)
    max_uti = -float('inf')
    boards = []
    for action in actions:
        boards.append(play_move(board,color,action[0],action[1]))
    QuickSort(boards,0,len(boards)-1,color)

    for board_tmp in boards:
        tmp = alphabeta_min_node(board_tmp,color,alpha,beta,level+1,limit)
        if tmp == None:
            continue
        max_uti = max(max_uti,tmp)
        if max_uti > beta or max_uti == beta:
            history[board] = max_uti
            return max_uti
        alpha = max(alpha,max_uti)
    history[board] = max_uti
    return max_uti


def select_move_alphabeta(board, color): 
    actions = get_possible_moves(board, color)
    alpha = -float('inf')
    beta = float('inf')
    v = -float('inf')
    max_v = -float('inf')

    limit = 6
    res = None

    for action in actions:
        v = alphabeta_min_node(play_move(board,color,action[0],action[1]),color,alpha,beta,1,limit)

        if v > max_v:
            max_v = v
            res = action
        alpha = max(alpha,v)
    if res:
        return res[0],res[1]
    else:
        return -1,-1

def QuickSort(arr,firstIndex,lastIndex,color):
    if firstIndex<lastIndex:
        divIndex=Partition(arr,firstIndex,lastIndex,color)
 
        QuickSort(arr,firstIndex,divIndex,color)       
        QuickSort(arr,divIndex+1,lastIndex,color)
    else:
        return
 
def Partition(arr,firstIndex,lastIndex,color):
    i=firstIndex-1
    for j in range(firstIndex,lastIndex):
        if compute_utility(arr[j],color) >= compute_utility(arr[lastIndex],color):
            i=i+1
            arr[i],arr[j]=arr[j],arr[i]
    arr[i+1],arr[lastIndex]=arr[lastIndex],arr[i+1]
    return i


####################################################
def run_ai():
    """
    This function establishes communication with the game manager. 
    It first introduces itself and receives its color. 
    Then it repeatedly receives the current score and current board state
    until the game is over. 
    """
    print("Minimax AI") # First line is the name of this AI  
    color = int(input()) # Then we read the color: 1 for dark (goes first), 
                         # 2 for light. 

    while True: # This is the main loop 
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input() 
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over. 
            print 
        else: 
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The 
                                  # squares in each row are represented by 
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)
                    
            # Select the move and send it to the manager 
            #movei, movej = select_move_minimax(board, color)
            movei, movej = select_move_alphabeta(board, color)
            print("{} {}".format(movei, movej)) 


if __name__ == "__main__":
    run_ai()
