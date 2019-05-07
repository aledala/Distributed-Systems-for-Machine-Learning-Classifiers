#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:33:57 2019

@author: shivangi
"""

import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pdb
import time
import argparse
import sklearn.model_selection as ms

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--min_neighbor', type=int, default=2)
parser.add_argument('--max_neighbor', type=int, default=15)

param = parser.parse_args()

###################### Data Loading ########################
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
#ld = load_digits()
#digits_x, digits_y = load_digits(return_X_y=True)
#x = pd.DataFrame(data=digits_x)
#y = pd.DataFrame(data=digits_y)
#x_train, x_test, y_train, y_test = train_test_split(x, digits_y, test_size=0.20, random_state=42)

#num_examples = x_train.shape[0] # training set size
#nn_input_dim = x_train.shape[1] # input layer dimensionality
#nn_output_dim = 2 # output layer dimensionality

#pdb.set_trace()

import scipy.io as io
file = io.loadmat('CSE812_Dataset-II.mat')
x_train = file['x_train']
x_test = file['x_test']
y_tr = file['y_train']
y_train = y_tr[0]
y_te = file['y_test']
y_test = y_te[0]

############################################################
cmd1 = 'KNC;'
algo = ['auto', 'ball_tree', 'kd_tree', 'brute']

for i in range(param.min_neighbor, param.max_neighbor):
    for j in range(0,len(algo)):
        cmd1 = cmd1 + str(i) + ' ' + algo[j] + ';'


###############################Classification on cmd1 ########################
best_acc = -1
best_cmd1 = ''
cmd = cmd1.split(';')
print(cmd[0])
if(cmd[0] == 'KNC'):
    for i in range(1,len(cmd)):
        print(i)
        print(cmd[i])
        print('\n')
        if(len(cmd[i])):
           n,a = cmd[i].split(' ')
           clf = KNeighborsClassifier(n_neighbors = int(n), algorithm = a)
           clf.fit(x_train, y_train)
           acc = clf.score(x_test, y_test)
           print(acc)
           if(acc > best_acc):
               best_acc = acc
               best_cmd1 = cmd[i]

print("Finished")
################################# Combine Results #############################
print("Accuracy: ", best_acc)

end = time.time()
print("Total time(in secs) is:",(end - start))