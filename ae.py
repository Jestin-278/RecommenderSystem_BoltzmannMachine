
# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
## u1.base and u1.test are training and test sets composed of 100k ratings in total (80-20 is the train-test split).
## There are multiple such files in order to allow for k-fold cross-validation if required
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int') # pytorch requires arrays and not dataframes
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')
## the columns now correspond to the users, movies, ratings and timeestamp (irrelevant)
## each row corresponds to a single rating

# Generating 2 matrices - one for training and the other for test
# The matrices will contain the users in rows, movies in columns and the cells filled with the corresponding ratings
# In R, this is equivalent to 'dcast'ing the movie column

## Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0]))) # taking the maximum of the highest User IDs in training and test sets
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Restructuring the data into an array (list of list as expected by FloatTensor function later) with users in lines and movies in columns
def convert(data):
    new_data = [] # this is the list of list for each user containing their ratings
    for id_users in range(1, nb_users + 1): # looping over all users
        id_movies = data[:,1][data[:,0] == id_users] # obtains all the movie ID for each user
        id_ratings = data[:,2][data[:,0] == id_users] # obtains the corresponding ratings
        ratings = np.zeros(nb_movies) # create a list of zeroes
        ratings[id_movies - 1] = id_ratings # replace zeroes for when there was a rating given
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module): # creates a child class for the Stacked AutoEncoder with Module class of the nn module as parent class through the process of inheritance
    def __init__(self, ):
        super(SAE, self).__init__() # to use the modules and methods from the nn class
        self.fc1 = nn.Linear(nb_movies, 20) # defines the full connection between the input layer (movie ratings) and 1st hidden layer
        self.fc2 = nn.Linear(20, 10) # the second hidden layer has 10 neurons and the first has 20
        self.fc3 = nn.Linear(10, 20) 
        self.fc4 = nn.Linear(20, nb_movies) # the third hidden layer has 20 neurons fully connnected to the output layer
        self.activation = nn.Sigmoid() # speicifies the activation function to activate the neurons when the observation goes into the network
    def forward(self, x): # defines the 3 encodings and 1 decoding that happens when the observation is forwarded into the network
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x # returns the predicted (reconstructed) ratings
sae = SAE() # initialize an object of the SAE class
criterion = nn.MSELoss() # specify the loss function for training
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # applies SGD in order to reduce the error at each epoch

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # to keep count of the users who rated at least one movie
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # convert input vector of one-dimension to two-dimensional batch
        target = input.clone() # create copy of the input for comparison; target will be updated during training
        if torch.sum(target.data > 0) > 0: # considers only users that rated atleast one movie
            output = sae(input) # create a vector of predicted ratings
            target.require_grad = False # to not apply SGD on target and only on input; avoids unnecessary computations
            output[target == 0] = 0 # include only non-zero values in the future computations of SGD for updating weights
            loss = criterion(output, target) 
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # average error for movies that were rated
            loss.backward() # instructs which way (add or subtract) to update weights; required only for trainings
            train_loss += np.sqrt(loss.data[0]*mean_corrector) # the unit of loss here is stars
            s += 1.
            optimizer.step() # specifies the magnitude by which weights will have to be updated
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0) # it is training_set because we are predicting ratings of the movies that the user hasn't watched yet, which are part of the training set
    target = Variable(test_set[id_user]).unsqueeze(0) # the real ratings
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))
