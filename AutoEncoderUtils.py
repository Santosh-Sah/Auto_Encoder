# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:03:51 2020

@author: Santosh Sah
"""

import pandas as pd
import numpy as np
import pickle

"""
Import training data set
"""
def importAutoEncoderTrainingDataset(autoEncoderTrainingDatasetFileName):
    
    autoEncoderDataset = pd.read_csv("ml-100k/u1.base", delimiter = '\t')
    
    autoEncoderTrainingDataset = np.array(autoEncoderDataset, dtype = 'int')
    
    return autoEncoderTrainingDataset

"""
Import testing data set
"""
def importAutoEncoderTestingDataset(autoEncoderTestingDatasetFileName):
    
    autoEncoderTestingDataset = pd.read_csv("ml-100k/u1.base", delimiter = '\t')
    
    autoEncoderTestingDataset = np.array(autoEncoderTestingDataset, dtype = 'int')
    
    return autoEncoderTestingDataset

"""
Converting the dataset into an array with users in lines and movies in columns. Rating will be in each cell
"""

def convertUsersInLineMoviesInColumns(dataset, maximnumNumberOfUsers, maximumNumberOfMovies):
    
    usersInLineMoviesInColumns = []
    
    for id_users in range(1, maximnumNumberOfUsers + 1):
        
        #creating a list of ratings given by the users to all the movies.
        #In the dataset the information available for rating given by all the users.
        #We need to get the list of rating for each user for every movie and create a list.
        id_rating = dataset[:, 2][dataset[:, 0] == id_users]
        
        #creating a list of movies to which rating given by users.
        #In the dataset the information available for movies given by all the users.
        #We need to get the list of movies.
        id_movies = dataset[:, 1][dataset[:, 0] == id_users]
        
        #There are many movies present in the list to which user did not give any rating.
        #We need to put zero if the user did not give any rating to the movies
        
        #Initialize the list of rating with zero
        ratings = np.zeros(maximumNumberOfMovies)
        
        #Fill the rating of the users given to movies. If rating not given then the value will be zero.
        #Rating start with 0 but numpy starts with 0 hence we need do -1
        ratings[id_movies - 1] = id_rating
        
        usersInLineMoviesInColumns.append(list(ratings))
    
    return usersInLineMoviesInColumns
    
"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
"""
read X_train from pickle file
"""
def readAutoEncoderXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readAutoEncoderXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
Save NumberOfusers as a pickle file.
"""
def saveNumberOfusers(numberOfusers):
    
    #Write NumberOfusers as a picke file
    with open("NumberOfusers.pkl",'wb') as NumberOfusers_Pickle:
        pickle.dump(numberOfusers, NumberOfusers_Pickle, protocol = 2)

"""
read NumberOfusers from pickle file
"""
def readNumberOfusers():
    
    #load NumberOfusers
    with open("NumberOfusers.pkl","rb") as NumberOfusers_pickle:
        numberOfusers = pickle.load(NumberOfusers_pickle)
    
    return numberOfusers

"""
Save NumberOfusers as a pickle file.
"""
def saveNumberOfMovies(numberOfMovies):
    
    #Write NumberOfMovies as a picke file
    with open("NumberOfMovies.pkl",'wb') as NumberOfMovies_Pickle:
        pickle.dump(numberOfMovies, NumberOfMovies_Pickle, protocol = 2)

"""
read NumberOfMovies from pickle file
"""
def readNumberOfMovies():
    
    #load NumberOfMovies
    with open("NumberOfMovies.pkl","rb") as NumberOfMovies_pickle:
        numberOfMovies = pickle.load(NumberOfMovies_pickle)
    
    return numberOfMovies
