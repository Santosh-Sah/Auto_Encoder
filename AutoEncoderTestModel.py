# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:48:50 2020

@author: Santosh Sah
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from AutoEncoderTrainModel import AutoEncoderArchitecture

from AutoEncoderUtils import (readAutoEncoderXTrain, readNumberOfusers, readNumberOfMovies, readAutoEncoderXTest)

def testAutoEncoder():
    
    numberOfusers = readNumberOfusers()
    numberOfMovies = readNumberOfMovies()
    
    autoEncoderTrainingSet = readAutoEncoderXTrain()
    autoEncoderTestingSet = readAutoEncoderXTest()
    
    autoEncoder = AutoEncoderArchitecture(numberOfMovies)
    
    criterion = nn.MSELoss()
            
    trainingLoss = 0
    numberOfUserRatedAtLeastOneMovie = 0.
        
    for id_user in range(numberOfusers):
            
        #getting the input vector whichhas all the ratings of all the movies given by a user.
        #we need put a second dimention here with the help of unsqueeze method as PyTorch do not take vector of single dimension.
        inputVector = Variable(autoEncoderTrainingSet[id_user]).unsqueeze(0)
            
        target = Variable(autoEncoderTestingSet[id_user]).unsqueeze(0)
            
        #we are putting here a if condition. This if condition is only to optimises the space. We are taking all those user who ratied
        #atlest one movie. We are free to use this if condition, if you have more memory then no need to use this if condition.
        if torch.sum(target.data > 0) > 0:
                
            #getting the output vector which will be generated after the processing of the input vector.
            #processing here means encoding and decoding 
            outputVector = autoEncoder.forward(inputVector)
                
            #we need to make sure the stochastic gradient descent must apply only on the input vector not on the cloned input vector.
            #do not comput the gradient with respect to the inputVectorCloned
            #it will save alot of computation and it will optimized the code.
            target.require_grad = False
                
            #we also want to compute the gradient on the non zero values.
            #we do not compute when the user did not rate any movies. 
            #it will be calculated on outputVector only
            #these observation will update its weights.These observation will not participate in the loss calculation
            outputVector[target == 0] = 0
                
            #calculate the loss. Loss will be calcuated with the help of predicted and actual value. 
            #in out case it is predicted ratings and actual ratings
            loss = criterion(outputVector, target)
                
            #no zero ratings
            #le-10 will make sure that the denominator must not be null
            #mean_corrector is avearage of the error but only for the movies which are rated
            mean_corrector = numberOfMovies / float(torch.sum(target.data > 0) + 1e-10)
                
            #backward method for the loss. It tells in which direction we should update the weights.
            #do we need to increase the weight or decrease the weight.
            loss.backward()
                
            #compute the RMSE
            trainingLoss += np.sqrt(loss.data * mean_corrector)
                
            numberOfUserRatedAtLeastOneMovie += 1.
                
    print("test  loss: " + str(trainingLoss / numberOfUserRatedAtLeastOneMovie))    

if __name__ == "__main__":
    testAutoEncoder()            
                
                