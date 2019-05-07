#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:42:24 2019

@author: shivangi
"""

import socket
import sys
import argparse
import numpy as np
import pdb
import time
from sklearn import svm

start_s1 = time.time()

#pdb.set_trace()

###################### Data Loading ########################
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
ld = load_digits()
digits_x, digits_y = load_digits(return_X_y=True)
x = pd.DataFrame(data=digits_x)
y = pd.DataFrame(data=digits_y)
x_train, x_test, y_train, y_test = train_test_split(x, digits_y, test_size=0.20, random_state=42)
###########################################################

parser = argparse.ArgumentParser()
parser.add_argument('--addr1', type=str,default='35.12.210.209')
parser.add_argument('--addr2', type=str, default='35.12.210.209')
parser.add_argument('--port1', type=int, default=2345)
parser.add_argument('--port2', type=int, default=2555)
parser.add_argument('--kernel', type=str, default='linear')
parser.add_argument('--min_g', type=int, default=-9)
parser.add_argument('--max_g', type=int, default=9)
parser.add_argument('--min_c', type=int, default=-2)
parser.add_argument('--max_c', type=int, default=14)
parser.add_argument('--min_d', type=int, default=1)
parser.add_argument('--max_d', type=int, default=4)

param = parser.parse_args()

print("Running: Client.py")

kernel_range = ['linear', 'rbf', 'poly']
C_range = np.logspace(param.min_c, param.max_c, 20)
gamma_range = np.logspace(param.min_g, param.max_g, 20)
degree_range = range(1,4)

cmd1 = 'svm;'
cmd2 = 'svm;'
cmd3 = 'svm;'
count = 0
for i in range(0,len(kernel_range)):
    for j in range(0, len(C_range)):
        for k in range(0, len(gamma_range)):
            for l in range(0, len(degree_range)):
                count = count+1
                if(count > 0 and count <= 1600):
                    cmd1 = cmd1 + kernel_range[i] + ' ' + str(C_range[j]) + ' ' + str(gamma_range[k]) + ' ' + str(degree_range[l]) + ';'
                elif(count >= 1600 and count <= 3200):
                    cmd2 = cmd2 + kernel_range[i] + ' ' + str(C_range[j]) + ' ' + str(gamma_range[k]) + ' ' + str(degree_range[l]) + ';'
                elif(count >= 3200):
                    cmd3 = cmd3 + kernel_range[i] + ' ' + str(C_range[j]) + ' ' + str(gamma_range[k]) + ' ' + str(degree_range[l]) + ';'

addr1 = param.addr1
port1 = param.port1
addr2 = param.addr2
port2 = param.port2

client1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Socket Successfully Created\n")

end_s1 = time.time()

start_p = time.time()

client1.connect((addr1, port1))
print("Client binded to: ", addr1)
print("/",port1)

byt1 = cmd2.encode()

print("Sending msg to Server-1")

client1.send(byt1)
print("Msg delivered to Server-1")

client2.connect((addr2, port2))
print("Client binded to: ", addr2)
print("/",port2)

byt2 = cmd1.encode()

print("Sending msg to Server-2")
client2.send(byt2)
print("Msg delivered to Server-2")

###############################Classification on cmd1 ########################
best_acc = -1
best_cmd1 = ''
cmd = cmd3.split(';')
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


print("Finished one set")
################################# Combine Results #############################

from_serv1 = client1.recv(10000)
from_serv2 = client2.recv(10000)
end_p = time.time()

start_s2 = time.time()

print("Accuracy from Server1: ", from_serv1.decode())
print("Accuracy from Server2: ", from_serv2.decode())
print("Accuracy from Server3: ", best_acc)

end_s2 = time.time()

print("Total Parallel time(in secs) is:",(end_p-start_p))
print("Total Serial time(in secs) is:",(end_s1-start_s1+ end_s2-start_s2)) 
print("Total time(in secs) is:",(end_s2-start_s1))


client1.close()
client2.close()






