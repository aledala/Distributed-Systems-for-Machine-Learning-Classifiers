#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:42:16 2019

@author: shivangi
"""
from sklearn import svm
import pdb

###################### Data Loading ########################
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
ld = load_digits()
digits_x, digits_y = load_digits(return_X_y=True)
x = pd.DataFrame(data=digits_x)
y = pd.DataFrame(data=digits_y)
x_train, x_test, y_train, y_test = train_test_split(x, digits_y, test_size=0.20, random_state=42)

################## Socket Connection #######################
import socket
import sys

print("Running: ", sys.argv[0])
addr1 = sys.argv[1]
port1 = int(sys.argv[2])

serv1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Socket Successfully Created\n")

serv1.bind((addr1, port1))
print("Socket binded to: ", addr1)
print("/",port1)

serv1.listen(5)
print("Socket is listening\n")

while True:
    conn, addr = serv1.accept()
    print("Got connection from: ", addr)
    data = conn.recv(1000000)
    commands = data.decode()
    
    cmd = commands.split(';')
    
    #pdb.set_trace()
    best_acc = -1
    best_cmd = ''
    if(cmd[0] == 'svm'):
        for i in range(1,len(cmd)-1):
            print(i)
            print(cmd[i]) 
            print('\n')
            k,c,g,d = cmd[i].split(' ')
            clf = svm.SVC(kernel=k, C = float(c), gamma=float(g), degree=float(d))
            clf.fit(x_train, y_train)
            acc = clf.score(x_test, y_test)
            print(acc)
            if(acc > best_acc):
                best_acc = acc
                best_cmd = cmd[i]
            
    
    serv1_msg = str(best_acc)
    serv1_byt = serv1_msg.encode()
    conn.send(serv1_byt)
    
    conn.close()
    serv1.close()
    print("Connection Closed")
    break






