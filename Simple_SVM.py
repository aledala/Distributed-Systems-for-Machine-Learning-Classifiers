#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:12:32 2019

@author: shivangi
"""

import numpy as np
from sklearn import svm
import pdb
import time
import argparse

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, default='linear')
parser.add_argument('--min_g', type=int, default=-9)
parser.add_argument('--max_g', type=int, default=9)
parser.add_argument('--min_c', type=int, default=-2)
parser.add_argument('--max_c', type=int, default=14)
parser.add_argument('--min_d', type=int, default=2)
parser.add_argument('--max_d', type=int, default=5)

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
############################################################

kernel_range = ['linear', 'rbf', 'poly']
C_range = np.logspace(param.min_c, param.max_c, 20)
gamma_range = np.logspace(param.min_g, param.max_g, 20)
degree_range = range(1,4)

cmd1 = 'svm;'
for i in range(0,len(kernel_range)):
    for j in range(0, len(C_range)):
        for k in range(0, len(gamma_range)):
            for l in range(0, len(degree_range)):
                cmd1 = cmd1 + kernel_range[i] + ' ' + str(C_range[j]) + ' ' + str(gamma_range[k]) + ' ' + str(degree_range[l]) + ';'

###############################Classification on cmd1 ########################
best_acc = -1
best_cmd1 = ''
cmd = cmd1.split(';')
print(cmd[0])
if(cmd[0] == 'svm'):
    for i in range(1,len(cmd)):
        print(i)
        print(cmd[i])
        print('\n')
        if(len(cmd[i])):
           k,c,g,d = cmd[i].split(' ')
           clf = svm.SVC(kernel=k, C = float(c), gamma=float(g), degree=float(d))
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





                