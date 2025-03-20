import numpy as np
import random as rm
import pandas as pd

TIME = 10

#each row is a timept 
start = [0.2, 0.2, 0.2, 0.2, 0.2]
states = [0,1,2,3,4]
transition_table = np.matrix([[0.86, 0.09, 0.01, 0.03, 0.01],[0.01, 0.75, 0.07, 0.08, 0.09],
                         [0.01, 0.02, 0.74, 0.21, 0.02],[0.21, 0.24, 0.22, 0.21, 0.12],
                         [0.01, 0.16, 0.05, 0.05, 0.73]])

#generates emission table for use in part 2
def emission():
    states = pd.read_csv ("genemarkers_states.tsv", sep = '\t', header=None)
    timepts = pd.read_csv ("genemarkers_timepoints.tsv", sep = '\t')
    for col_name, col_values in states.items():
        print(f"Column name: {col_name}")
        print(col_values.values)
        col = col_values.values
        sum = 0
        for i in range(len(col)):
            sum += col[i]
        for i in range(len(col)):
            col[i] = col[i] / sum


def basic_model(trans, states, init):
    prob = np.zeros((TIME + 1,len(states)),dtype=float)
    prev = np.zeros((TIME + 1,len(states)),dtype=float)
    for s in states:
        prob[0][s] = init[s]
    
    for t in range(1, TIME + 1):
        for s in states:
            for r in states:
                new_prob = prob[t-1, r] * trans[r,s]
                if new_prob > prob[t,s]:
                    prob[t][s] = new_prob
                    prev[t][s] = r
    path = [0] * (TIME + 1)
    path[TIME] = 2
    #print(prev)
    for t in range(TIME - 1, 0, -1):
        path[t] = int(prev[t+1][int(path[t+1])])
    #print(prob)
    return path

def viterbi(states, init, trans, emit, observe):
    prob = np.zeros((len(observe),len(states)),dtype=float)
    prev = np.zeros((len(observe),len(states)),dtype=float)
    for s in states:
        prob[0][s] = init[s] * emit[s][observe[0]]
    
    for t in range(1, len(observe)):
        for s in states:
            for r in states:
                new_prob = prob[t-1][r] * trans[r][s] * emit[s][observe[t]]
                if new_prob > prob[t][s]:
                    prob[t][s] = new_prob
                    prev[t][s] = r

    path = [0] * len(observe)
    path[len(observe) - 1] = prob[len(observe) - 1][2]
    for t in range(len(observe) - 2, 0, -1):
        path[t] = int(prev[t+1][int(path[t+1])])
    return path

def main():
    print(basic_model(transition_table, states, start))
    emission()


if __name__ == '__main__':
    main()