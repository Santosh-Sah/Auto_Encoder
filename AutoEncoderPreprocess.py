# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:05:21 2020

@author: Santosh Sah
"""
import torch
from AutoEncoderUtils import (importAutoEncoderTrainingDataset, importAutoEncoderTestingDataset, convertUsersInLineMoviesInColumns,
                                   saveTrainingAndTestingDataset, saveNumberOfusers, saveNumberOfMovies)

def preprocess():
    
    autoEncoderTrainingDataset = importAutoEncoderTrainingDataset("ml-100k/u1.base")
    autoEncoderTestingDataset = importAutoEncoderTestingDataset("ml-100k/u1.test")
    
    # Getting the number of users and movies
    numberOfUsers = int(max(max(autoEncoderTrainingDataset[:,0]), max(autoEncoderTestingDataset[:,0])))
    numberOfMovies = int(max(max(autoEncoderTrainingDataset[:,1]), max(autoEncoderTestingDataset[:,1])))
    
    autoEncoderTrainingSet = convertUsersInLineMoviesInColumns(autoEncoderTrainingDataset, numberOfUsers, numberOfMovies)
    autoEncoderTestinggSet = convertUsersInLineMoviesInColumns(autoEncoderTestingDataset, numberOfUsers, numberOfMovies)
    
    #converting training data into torch tensors
    autoEncoderTrainingSet = torch.FloatTensor(autoEncoderTrainingSet)
    
    #converting testing data into torch tensors
    autoEncoderTestinggSet = torch.FloatTensor(autoEncoderTestinggSet)
        
    saveTrainingAndTestingDataset(autoEncoderTrainingSet, autoEncoderTestinggSet)
    
    saveNumberOfusers(numberOfUsers)
    
    saveNumberOfMovies(numberOfMovies)

    
if __name__ == "__main__":
    preprocess()