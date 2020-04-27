# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 07:29:55 2020

@author: Santosh Sah
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from AutoEncoderUtils import (readAutoEncoderXTrain, readNumberOfusers, readNumberOfMovies)

#creating the architecture of the auto encoder neural network
class AutoEncoderArchitecture(nn.Module):
    
    def __init__(self, numberOfMovies):
        
        #instantialte all the method and properties of the super class.
        super(AutoEncoderArchitecture, self).__init__()
        
        #full connection between input vector of features and the first hidden layer.
        #autoencoder encoding the input vector into shorter vector that will take place in first hidden layer.
        #using a class from nn module which will represent the full connection of the input vector and first encoded vectors.
        #first parameter of the Linear class is number of input features, in our case the number of feature is number of movies
        #second parameter of Linear class will be the number of neuron in the first hidden layer. Based on the experiment we will taking it 20.
        #number of neurons in the hidden layer can be optiimsed and we can build several model based on the different number of neuron in the hidden layer.
        #neuron in the auto encoder represent the features. Features like actor, types of the movies, directors etc. If a new user come and if he gave
        #good rating to the horror typeof movies then based on that some neuron i.e., features gets activated.
        self.firstFullConnection = nn.Linear(numberOfMovies, 20)
        
        #second full connection layer.
        #here we are maling the full connection between first hidden layer and the second hidden layer.
        #first argument of the linear class will be the number of neurons in the first hidden layer. Here it is 20
        #second parameter will be the number of neuron in the second hidden layer. Based on the experiment we are taking it as 10.
        self.secondFullConnection = nn.Linear(20, 10)
        
        #third hidden layer
        #from this third hidden layer we are not encoding more, from here we are start to decoding. Hence we need to make it symmetrical with initial input vector.
        #to make it symmetrical we need to put the number of neuron in this hidden layer same as the number of neuron in the first hidden layer.
        self.thirdFullConnection = nn.Linear(10, 20)
        
        #last full connection we need to make for decoding.
        #here the first parameter has to be the number of nuerons in the third layer. Here it will be 20
        #here we are also construction the output vector. In auto encoder the output vector must be same as input vector.
        self.fourthFullConnection = nn.Linear(20, numberOfMovies)
        
        #define the activation function. If some one liked the movie then some neuron gets activated. 
        #activation of the neuron will be based on the activation function.
        #in out case the activation function is sigmoid.
        self.activation = nn.Sigmoid()
    
    #encoding and decoding in the networks
    #it will give the vector of the predicted rating. With these predictd rating we can compare with the actual ratings i.e., the input vector.   
    #here we are encoding and decoding the input vector twice to get the predicted ratings.
    def forward(self, inputVector):
        
        #first encoding of the input vector to the first hidden layer of 20 neurons.
        #the sigmoid activation function will activate the neuron of the first hidden layer.
        #for encoding and decoding it will modify inputVector
        inputVector = self.activation(self.firstFullConnection(inputVector))
        
        #second encoding of the modified input vector to the second hidden layer.
        #the sigmoid activation function will activate the neuron of the second hidden layer.
        #for encoding and decoding it will modify inputVector.
        inputVector = self.activation(self.secondFullConnection(inputVector))
        
        #decoding the modified input vector to the third hidden layer.
        #the sigmoid activation function will activate the neuron of the third hidden layer.
        #for encoding and decoding it will modify inputVector.
        inputVector = self.activation(self.thirdFullConnection(inputVector))
        
        #decoding the modified input vector to the fourth hidden layer.
        #we will not use the activation function here because in this layer no activation of neuron will happen.
        #for encoding and decoding it will modify inputVector.
        #it will give the output vector.
        outputVector = self.fourthFullConnection(inputVector)
        
        return outputVector

def trainAutoEncoder():
    
    numberOfusers = readNumberOfusers()
    numberOfMovies = readNumberOfMovies()
    
    autoEncoderTrainingSet = readAutoEncoderXTrain()
    
    autoEncoder = AutoEncoderArchitecture(numberOfMovies)
    
    criterion = nn.MSELoss()
    
    #defining the optimiser. each optimiser has dufferent class.
    #decay is used to reduce the learning rate after every few epochs to regulate the convergent.
    optimiser = optim.RMSprop(autoEncoder.parameters(), lr = 0.01, weight_decay = 0.5)
    
    numberOfEpoch = 200
    
    for epoch in range(1, numberOfEpoch + 1):
        
        trainingLoss = 0
        numberOfUserRatedAtLeastOneMovie = 0.
        
        for id_user in range(numberOfusers):
            
            #getting the input vector whichhas all the ratings of all the movies given by a user.
            #we need put a second dimention here with the help of unsqueeze method as PyTorch do not take vector of single dimension.
            inputVector = Variable(autoEncoderTrainingSet[id_user]).unsqueeze(0)
            
            inputVectorCloned = inputVector.clone()
            
            #we are putting here a if condition. This if condition is only to optimises the space. We are taking all those user who ratied
            #atlest one movie. We are free to use this if condition, if you have more memory then no need to use this if condition.
            if torch.sum(inputVectorCloned.data > 0) > 0:
                
                #getting the output vector which will be generated after the processing of the input vector.
                #processing here means encoding and decoding 
                outputVector = autoEncoder.forward(inputVector)
                
                #we need to make sure the stochastic gradient descent must apply only on the input vector not on the cloned input vector.
                #do not comput the gradient with respect to the inputVectorCloned
                #it will save alot of computation and it will optimized the code.
                inputVectorCloned.require_grad = False
                
                #we also want to compute the gradient on the non zero values.
                #we do not compute when the user did not rate any movies. 
                #it will be calculated on outputVector only
                #these observation will update its weights.These observation will not participate in the loss calculation
                outputVector[inputVectorCloned == 0] = 0
                
                #calculate the loss. Loss will be calcuated with the help of predicted and actual value. 
                #in out case it is predicted ratings and actual ratings
                loss = criterion(outputVector, inputVectorCloned)
                
                #no zero ratings
                #le-10 will make sure that the denominator must not be null
                #mean_corrector is avearage of the error but only for the movies which are rated
                mean_corrector = numberOfMovies / float(torch.sum(inputVectorCloned.data > 0) + 1e-10)
                
                #backward method for the loss. It tells in which direction we should update the weights.
                #do we need to increase the weight or decrease the weight.
                loss.backward()
                
                #compute the RMSE
                trainingLoss += np.sqrt(loss.data * mean_corrector)
                
                numberOfUserRatedAtLeastOneMovie += 1.
                
                #use optimiser to update the weights.
                #optimiser.step will decide the intensity of the updates
                optimiser.step()
                
    print("epoch: " + str(epoch) + " loss: " + str(trainingLoss / numberOfUserRatedAtLeastOneMovie))    
    #loss 0.9108

if __name__ == "__main__":
    trainAutoEncoder()            
                
                