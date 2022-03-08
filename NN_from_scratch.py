# -*- coding: utf-8 -*-
"""
Data downloaded from Assignment 4, Andrew NG Intro to ML (adapted to python)

Github Page with Assignment and theory:
https://github.com/dibgerge/ml-coursera-python-assignments/tree/master/Exercise4

Implementing the feedforward/backpropagation algorithm and for a 2 later neural network
and applying it to the MNIST dataset. 
"""


import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import time

"""the first 4 functions are from the utils file from github, 
they are there to check our code"""

def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        # Display Image
        h = ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                      cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')


def debugInitializeWeights(fan_out, fan_in):
    """
    Initialize the weights of a layer with fan_in incoming connections and fan_out outgoings
    connections using a fixed strategy. This will help you later in debugging.

    Note that W should be set a matrix of size (1+fan_in, fan_out) as the first row of W handles
    the "bias" terms.
    Parameters
    ----------
    fan_out : int
        The number of outgoing connections.
    fan_in : int
        The number of incoming connections.
    Returns
    -------
    W : array_like (1+fan_in, fan_out)
        The initialized weights array given the dimensions.
    """
    # Initialize W using "sin". This ensures that W is always of the same values and will be
    # useful for debugging
    W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
    W = W.reshape(fan_out, 1+fan_in, order='F')
    return W

def checkNNGradients(nnCostFunction, lambda_=0):

    """
    Creates a small neural network to check the backpropagation gradients. It will output the
    analytical gradients produced by your backprop code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient computations should result in
    very similar values.

    Parameters
    ----------
    nnCostFunction : func
        A reference to the cost function implemented by the student.
    lambda_ : float (optional)
        The regularization parameter value.
    """
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = np.arange(1, 1+m) % num_labels
    # print(y)
    # Unroll parameters
    nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])
    # short hand for cost function
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                        num_labels, X, y, lambda_)
    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)
    # Visually examine the two gradient computations.The two columns you get should be very similar.
    print(np.stack([numgrad, grad], axis=1))
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')
    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
    # should be less than 1e-9.
    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)
    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          'Relative Difference: %g' % diff)

def computeNumericalGradient(J, theta, e=1e-4):
    """    
    Computes the gradient using "finite differences" and gives us a numerical estimate of the
    gradient.

    Params
    J : func
        The cost function which will be used to estimate its numerical gradient.

    theta : array_like
        The one dimensional unrolled network parameters. The numerical gradient is computed at
         those given parameters.

    e : float (optional)
        The value to use for epsilon for computing the finite difference.

    Notes
    -----
    The following code implements numerical gradient checking, and
    returns the numerical gradient. It sets `numgrad[i]` to (a numerical
    approximation of) the partial derivative of J with respect to the
    i-th input argument, evaluated at theta. (i.e., `numgrad[i]` should
    be the (approximately) the partial derivative of J with respect
    to theta[i].)
    """

    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))

    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*e)

    return numgrad

def randInitializeWeights(L_in,L_out,epsilon_init=0.12):
    
    """
    Randomly initialize the weights of a layer in a neural network.

    Parameters
    ----------
    L_in : int
        Number of incomming connections.

    L_out : int
        Number of outgoing connections. 

    epsilon_init : float, optional
        Range of values which the weight can take from a uniform 
        distribution.

    Returns
    -------
    W : array_like
        The weight initialiatized to random values.  Note that W should
        be set to a matrix of size(L_out, 1 + L_in) as
        the first column of W handles the "bias" terms.
    
    Instructions
    ------------
    Initialize W randomly so that we break the symmetry while training
    the neural network. Note that the first column of W corresponds 
    to the parameters for the bias unit.
    """
    
    return (np.random.rand(L_out,L_in+1)*2*epsilon_init-epsilon_init)

def sigmoid(matrix):
    return (1/(1+np.exp(-matrix)))

def sigmoidgradient(matrix):
    g=sigmoid(matrix)
    return g*(1-g)

def predictNN(Theta1,Theta2,X):
    """
    Predict the label of an input given a trained neural network
    Outputs the predicted label of X given the trained weights of a neural
    network(Theta1, Theta2)
    
    Parameters
    ----------
    Theta1 : Weights for first layer. 
             Has shape (hidden layer size, input layer size + 1)
    Theta2 : Weights for second layer. 
             Has shape (hidden layer size + 1, output layer size)
    X : Input matrix with rows corresponding to data points. 
        Has shape (# of data points, # of features)

    Returns
    -------
    p : Array of numbers that correspond to the highest probability outputted
        by the NN. 
    """
    
    
    m = len(X[:,0])
    #this adds the bias units
    
    ones = np.ones((m,1))
    X = np.hstack((ones,X))
   
    x = X.T #this makes each example a vector! as in intended in NN's
    
    output1 = sigmoid(np.dot(Theta1,x)) #the output of the first hidden layers
    
    ones_o1 = np.ones((1,len(output1[0]))) #this is to add the bias later at the top
    
    output1 = np.vstack((ones_o1,output1))

    #final output layer each row lists the prob for data points corresponding to label

    output_final = sigmoid(np.dot(Theta2,output1))     
    p = np.zeros((1,len(output_final[0])))
    
    p = np.argmax(output_final,axis=0)
    
    return p

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    
    """
    Implements the neural network cost function and gradient for a two layer neural 
    network which performs classification. 

    Parameters
    ----------
    nn_params : array_like
        The parameters for the neural network which are "unrolled" into 
        a vector. This needs to be converted back into the weight matrices Theta1
        and Theta2.

    input_layer_size : int
        Number of features for the input layer. 

    hidden_layer_size : int
        Number of hidden units in the second layer.

    num_labels : int
        Total number of labels, or equivalently number of units in output layer. 

    X : array_like
        Input dataset. A matrix of shape (m x input_layer_size).

    y : array_like
        Dataset labels. A vector of shape (m,).

    lambda_ : float, optional
        Regularization parameter.

    Returns
    -------
    J : float
        The computed value for the cost function at the current weight values.

    grad : array_like
        An "unrolled" vector of the partial derivatives of the concatenatation of
        neural network weights Theta1 and Theta2.
    """
    
    #the first step is for us to roll our nn_params into matrices
    # Theta1 has size 25 x 401, in other words, hidden layer sizex(inputsize+1)
    
    Theta1=np.reshape(nn_params[:(hidden_layer_size*(input_layer_size+1))],(hidden_layer_size,input_layer_size+1))
    
    # Theta2 has size 10 x 26, in other words, output layer sizexhiddenlayersize+1
    
    Theta2=np.reshape(nn_params[(hidden_layer_size*(input_layer_size+1)):],(num_labels,hidden_layer_size+1))
    
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    J=0
    m=len(y)
    
    x=X.T
    
    ones=np.ones((1,len(x[0])))
    x=np.vstack((ones,x)) #each column of x is now a data point! the first row is the bias element! 
    
    for i in range(len(x[0])):
        y_v=np.zeros((num_labels,1))
        y_v[y[i]][0]=1 #initializes to the value of y that we need to use! 

        #feedforward
        a1=x[:,i]
        a1=a1[:,np.newaxis]
        
        z2=np.dot(Theta1,a1)
        
        z2=np.vstack(([1],z2))
        
        a2=sigmoid(z2)
        a2[0][0]=1
                
        z3=np.dot(Theta2,a2)
        
        a3=sigmoid(z3)
        
        #backpropagate
        delta_3=a3-y_v
        
        delta_2=np.dot(Theta2.T,delta_3)*sigmoidgradient(z2)
        
        Theta1_grad=Theta1_grad+np.dot(delta_2[1:],a1.T)
        Theta2_grad=Theta2_grad+np.dot(delta_3,a2.T)
        
        J_temp=-np.dot(y_v.T,np.log(a3))-np.dot((1-y_v).T,np.log(1-a3))
        J_temp=J_temp/m
        
        J=J+J_temp
     
    Theta1_reg=np.sum(np.power(Theta1[:,1:],2))#all columns except the first
    Theta2_reg=np.sum(np.power(Theta2[:,1:],2))
    
    #this part will add the regularization terms, I suspect, they are just the sum of the values squared
    J=J+(Theta1_reg+Theta2_reg)*lambda_/(2*m)
    
    Theta1_grad=Theta1_grad/m
    Theta1_grad[:,1:]=Theta1_grad[:,1:]+(lambda_/m)*Theta1[:,1:] #adds the regularization term
    Theta2_grad=Theta2_grad/m
    Theta2_grad[:,1:]=Theta2_grad[:,1:]+(lambda_/m)*Theta2[:,1:] #adds the regularization term
    
    grad=np.concatenate([Theta1_grad.ravel(),Theta2_grad.ravel()])
    
    return J[0][0],grad

"""Execution code starts here --------------------------------------------"""
start_time=time.time()

#  training data stored in arrays X, y (matlab)
data = loadmat('ex4data1.mat')
X, y = data['X'], data['y'].ravel()

m = len(y) # number of total data points (or images) 

"""
# Uncomment block to randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]
displayData(sel)
"""



input_layer_size  = 400 # 20x20 Input Images of Digits

hidden_layer_size=25

num_labels = 10 # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)



y[y == 10] = 0 # set the zero digit to 0, rather than its mapped 10 in this dataset

weights = loadmat('ex4weights.mat') # get the model weights from the dictionary

Theta1, Theta2 = weights['Theta1'], weights['Theta2']
#this will cyclicly permutate the columns of theta 2 because of the matlab indexing
Theta2 = np.roll(Theta2, 1, axis=0)

nn_params=np.concatenate([Theta1.flatten(),Theta2.flatten()]) #these are our unrolled parameters
#the above values were used for checking things, now we will do new things.

initial_Theta1=randInitializeWeights(400,25)
initial_Theta2=randInitializeWeights(25,10)

initial_nn_params=np.concatenate([initial_Theta1.flatten(),initial_Theta2.flatten()])

#cost,grad=nnCostFunction(initial_nn_params,input_layer_size,25,num_labels,X, y,lambda_=3)

#this part is running the optimization of the weights
options= {'maxiter': 300}

lambda_ = 2

# instead of passing everything to the optimizer, you just need to pass the inpul paramethers
costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                        hidden_layer_size,
                                        num_labels, X, y, lambda_)

# Now, costFunction is a function that takes in only one argument
# (the neural network parameters)
res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

# get the solution of the optimization
nn_params = res.x
#optimization_successs = res.success         

# Obtain Theta1 and Theta2 back from nn_params, converting to matrix again
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))

#makes the prediction using NN
pred = predictNN(Theta1,Theta2,X)

print('Training Set Accuracy: %f' % (np.mean(pred == y.flatten()) * 100))
#print(cost)

print('Time:',time.time()-start_time)
