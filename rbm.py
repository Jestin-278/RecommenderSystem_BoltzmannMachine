
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

# Converting the data into an array (list of list as expected by FloatTensor function later) with users in lines and movies in columns
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

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh): # defines the weights and the bias of the RBM once the class is made
        self.W = torch.randn(nh, nv) # weight is a matrix of torch tensors with nh columns and nv rows containing values that follow a normal distribution
        self.a = torch.randn(1, nh) # bias for the probabilities of the hidden nodes given the visible nodes
        self.b = torch.randn(1, nv) # bias for the probabilities of the visible nodes given the hidden nodes
    def sample_h(self, x): # samples the probabilities of the hidden nodes being activated given the values of the visible nodes
        wx = torch.mm(x, self.W.t()) # obtain the product of the input vector x and the transpose of the tensor of weights W
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v) # returns a vector of nh elements containing the probability of each hidden node being activated, along with its bernoulli's sampling
    def sample_v(self, y): # samples the probabilities of the visible nodes being activated given the values of the hidden nodes
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk): # takes as inputs the input vector of ratings by a user, the visible node obtained after k iterations, the vector of probabilities of hidden nodes at the first iteration, and the vector of probabilities of hidden nodes after k samplings given the values of the visible nodes
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t() # weights will be updated towards the direction of maximumn log likelihood
        self.b += torch.sum((v0 - vk), 0) # biases will be updated towards the direction of maximumn log likelihood
        self.a += torch.sum((ph0 - phk), 0)
# Creating an RBM object
nv = len(training_set[0]) # number of visible nodes is the number of features in the training/test set matrix after dcasting, ie, number of movies
nh = 100 # arbitrary choice for the number of features to be detected
batch_size = 100 # number of observations after which the weights and biases in the RBM class will be adjusted
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # counter (of type float) to normalize the train loss
    for id_user in range(0, nb_users - batch_size, batch_size): # looping through different batches
        vk = training_set[id_user:id_user+batch_size] # input batch containing ratings of all the users in the batch
        v0 = training_set[id_user:id_user+batch_size] # target, which unllke the input batch will not be updated during contrastive divergence
        ph0,_ = rbm.sample_h(v0) # initial probabilities
        for k in range(10):  # for the k steps (random walk) of contrastive divergence
            _,hk = rbm.sample_h(vk) # walk from the visible node to the hidden node
            _,vk = rbm.sample_v(hk) # walk from the hidden node to the visible node
            vk[v0<0] = v0[v0<0] # keep the original -1 ratings as is, i.e., make sure the training was not done for ratings that were not existent
        phk,_ = rbm.sample_h(vk) # obtain vector of probabilities of hidden nodes after k samplings given the values of the visible nodes
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])) # simple difference between predicted and actual
        s += 1. # increment the counter
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))
