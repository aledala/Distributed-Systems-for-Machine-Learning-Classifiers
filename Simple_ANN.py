#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 02:18:30 2019

@author: shivangi
"""

import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import pdb
import time
import argparse

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--min_hidden', type=int, default=100)
parser.add_argument('--max_hidden', type=int, default=200)
parser.add_argument('--min_learning', type=int, default=0.005)
parser.add_argument('--max_learning', type=int, default=0.05)

param = parser.parse_args()

###################### Data Loading ########################

from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
ld = load_digits()
digits_x, digits_y = load_digits(return_X_y=True)
x = pd.DataFrame(data=digits_x)
y = pd.DataFrame(data=digits_y)
x_train, x_test, y_train, y_test = train_test_split(x, digits_y, test_size=0.20, random_state=42)

#import scipy.io as io
#file = io.loadmat('CSE812_Dataset-II.mat')
#x_train = file['x_train']
#x_test = file['x_test']
#y_tr = file['y_train']
#y_train = y_tr[0]
#y_te = file['y_test']
#y_test = y_te[0]

num_examples = x_train.shape[0] # training set size
nn_input_dim = x_train.shape[1] # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality

############################################################
cmd1 = 'ANN;'
activ = ['identity', 'logistic', 'tanh', 'relu']
solv = ['lbfgs', 'sgd', 'adam']


for i in range(0,len(activ)):
    for j in range(0,len(solv)):
        hid = param.min_hidden
        while (param.max_hidden >= hid):
            ler = param.min_learning
            while(param.max_learning >= ler):
                cmd1 = cmd1 + str(hid) + ' ' + str(ler) + ' ' + activ[i] + ' ' + solv[j] +';'
                ler = ler + 0.005
            hid = hid+10
    

###############################Classification on cmd1 ########################
best_acc = -1
best_cmd1 = ''
cmd = cmd1.split(';')
print(cmd[0])
if(cmd[0] == 'ANN'):
    for i in range(1,len(cmd)):
        print(i)
        print(cmd[i])
        print('\n')
        if(len(cmd[i])):
           h,l,a,s = cmd[i].split(' ')
           clf = MLPClassifier(hidden_layer_sizes = [int(h)-50, int(h), int(h)+50], 
                                                     learning_rate_init=float(l), activation=a, solver=s)
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


