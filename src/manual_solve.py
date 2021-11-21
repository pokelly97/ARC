#!/usr/bin/python

import os, sys
import json
import numpy as np
import re
from collections import Counter
from itertools import product

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
#def solve_f25ffba3(x):
#    out_arr = x.copy()
#    shape = x.shape
#    half_height = shape[0] / 2
#    for j in range(shape[1]):
#        for i in range(int(half_height)):
#            out_arr[i, j] = x[shape[0]-1-i, j]
#    return out_arr   
    
#def solve_9af7a82c(x):
#    count_dict = Counter()
#    shape = x.shape
#    for i in range(shape[0]):
#        for j in range(shape[1]):
#            count_dict[x[i, j]] += 1
#    shape_1 = len(count_dict)
#    shape_0 = max(count_dict.values())
#    out_arr = np.zeros((shape_0, shape_1), int)
#    for j, item in enumerate(sorted(count_dict.items(), 
#                         key=lambda x: count_dict[x[0]], 
#                         reverse=True)):
#        out_arr[:item[1], j] = item[0]
#    
#    return out_arr

def perform_search(x, center):
    height, width = x.shape
    red = 2
    j, i = center
    bounds = [j, j, i, i]
    
    neighbourhood = [(row, col) for row in range(j, j+3) for col in range(i-2, i+3) if (row in range(height) and col in range(width))]
    checked = [(index, x[index]==red) for index in neighbourhood]
    
    red_squares = [tup[0] for tup in checked if tup[1]==True]
    bounds[1] = max([index[0] for index in red_squares])
    bounds[2] = min([index[1] for index in red_squares])
    bounds[3] = max([index[1] for index in red_squares]) 
            
    return bounds
    
    
def solve_36fdfd69(x):
    height, width = x.shape
    out_arr = x.copy()
    
    counts = np.bincount(x.flatten())
    colour_counts = counts[1:]
    majority_colour = np.argmax(colour_counts)
    red = 2
    yellow = 4
    
    skip = []
    
    for j in range(x.shape[0]):
        for i in range(x.shape[1]):
            if x[j, i] == red:
                center = (j, i)
                bounds = [j, j, i, i]
                for iteration in range(2):
                    corner_bounds = [bounds]
                    corners = set(product(bounds[:2], bounds[2:]))     
                    for corner in corners:
                        corner_bounds.append(perform_search(x, center))
                    corner_array = np.array(corner_bounds)
                    bounds[1] = max(corner_array[:, 1])
                    bounds[2] = min(corner_array[:, 2])
                    bounds[3] = max(corner_array[:, 3])
                
                    
                to_fill = [(row, col) for row in range(bounds[0], bounds[1]+1) for col in range(bounds[2], bounds[3]+1) if x[row, col] != red]
                
                for index in to_fill:
                    out_arr[index] = yellow
    
    
    return out_arr


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

