import numpy as np
import random as rm
import pandas as pd

TIME = 10

#each row is a timept 
start = [0.2, 0.2, 0.2, 0.2, 0.2]
states = [0,1,2,3,4]
observation= [0,1,2,3,4,5,6,7,8,9]
transition_table = np.matrix([[0.86, 0.09, 0.01, 0.03, 0.01],[0.01, 0.75, 0.07, 0.08, 0.09],
                         [0.01, 0.02, 0.74, 0.21, 0.02],[0.21, 0.24, 0.22, 0.21, 0.12],
                         [0.01, 0.16, 0.05, 0.05, 0.73]])

def scale_distances(distances):
    min_distance = min(distances)
    max_distance = max(distances)

    if min_distance == max_distance:
      return [0.0] * len(distances)

    return [(d - min_distance) / (max_distance - min_distance) for d in distances]

#generates emission table for use in part 2
def emission():
    states = pd.read_csv ("genemarkers_states.tsv", sep = '\t', header=None)
    timepts = pd.read_csv ("genemarkers_timepoints.tsv", sep = '\t', )
    states = np.asmatrix(states.to_numpy())
    timepts = np.asmatrix(timepts.to_numpy())
    mat = np.zeros((5, 10),dtype=float)
    #print(states, timepts)
    shape = states.shape
    shape2 = timepts.shape
    print(shape)

    #normalize both matrices
    for i in range(shape[0]):
        sumn = 0
        for j in range(1, shape[1]):
            sumn += states[i,j]
        for j in range(1,shape[1]):
            states[i,j] = states[i,j] / sumn
    for i in range(shape2[0]):
        sumn = 0
        for j in range(1, shape2[1]):
            sumn += timepts[i,j]
        for j in range(1,shape[1]):
            timepts[i,j] = timepts[i,j] / sumn
    #now we calculate average distances
    
    for t in range(shape2[0]):
        dist = 0
        distances = [0] * 5
        for k in range(shape[0]):
            for j in range(1, shape2[1]):
                dist += (timepts[t,j] - states[k,j])
            dist = abs(dist) / shape2[1] 
            #absolute averaged distance for this state/observation
            distances[k] = dist
            distances = scale_distances(distances)
        for d in range(len(distances)):
            mat[d,t] = distances[d]
    return mat


        



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
                new_prob = prob[t-1][r] * trans[r,s] * emit[s,observe[t]]
                if new_prob > prob[t,s]:
                    prob[t][s] = new_prob
                    prev[t][s] = r

    path = [0] * len(observe)
    path[len(observe) - 1] = prob[len(observe) - 1][2]
    for t in range(len(observe) - 2, 0, -1):
        path[t] = int(prev[t+1][int(path[t+1])])
    print(prob)
    return path

def main():
    print(basic_model(transition_table, states, start))
    print(viterbi(states, start, transition_table, emission(), observation))


if __name__ == '__main__':
    main()