# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:16:10 2023

@author: monk
"""
import os

def mkdir(path):
 
    folder = os.path.exists(path)
 
    if not folder:                   
        os.makedirs(path)            
        print ("---  The directory is correct...  ---")
 
    else:
        print ("---  The directory has been created correctly!  ---")

data_id = [3, 4, 7, 8]
for i in data_id:
    file_accuracy = "./save/PEMS0{}/accuracy/".format(i)
    file_data = "./save/PEMS0{}/data/".format(i)
    file_log = "./save/PEMS0{}/log/".format(i)
    file_loss = "./save/PEMS0{}/loss/".format(i)
    mkdir(file_accuracy)
    mkdir(file_data)
    mkdir(file_log)
    mkdir(file_loss)
