import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import gzip as gz


numpts = 10000 #total number of points in file is 767928, but can be reduced for now, for training

#upload data and labels from files
labels = np.zeros((numpts,18))
data = np.zeros((numpts,1000,4,1))

verbose = True
with gz.open('hema_model_jason_only_oct17_idr_K562_GM12878_CV_corePeakSize_500_totalPeakSize_1000_neg_flank_offset_500_Sequential_1D_model_binaryPeaks.txt.gz', 'rb') as f:
    file_content = f.readline()
    file_content = str(file_content[0:-3])
    file_content = file_content.split('\\t')
    file_content[0] = file_content[0][2:-1]
    file_content[-1] = file_content[-1][0:-2]
    
    for i in range(numpts):
        file_content = f.readline()
        file_content = str(file_content[0:-1])
        file_content = file_content.split('\\t')
        file_content[0] = file_content[0][2:-1]
        file_content[-1] = file_content[-1][0:-1]

        labels[i][:] = [int(cont) for cont in file_content[4:]]
            
        if np.mod(i,int(numpts/10))==0 and verbose:
            print(i,len(file_content[4:]),file_content[0], labels[i])
        
with open('data/chr1_bp.txt','r') as f:
    genDict = {'n':4,'a':0, 'c':1,'t':2, 'g':3}
    for i in range(numpts):
        file_content = f.readline()
        file_content = f.readline()
        geneseq = list(file_content)
        geneseq = geneseq[0:-1]
        basecount = 0
        for base in geneseq:
            
            genInd = genDict[base.lower()]
            if genInd!=4:
                data[i,basecount,genInd,0] = 1
            basecount += 1


####################################################################################
####################    BEGIN MODEL   ##############################################
####################################################################################
#code taken largely from Coursera class 4 on CNN's in tensorflow

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    '''create placeholders for tensorflow X and Y'''
    X = tf.placeholder("float",shape=(None,n_H0,n_W0,n_C0))
    Y = tf.placeholder("float",shape=(None,n_y))
    
    return X, Y

def initialize_parameters(initP=0):
    '''create CNN filters.  If initP is present,
        use the given values, else use xavier initialization'''
    if initP !=0:
        W1 = initP("W1")
        W2 = initP("W2")
    else:
        W1 = tf.get_variable("W1",[4,4,1,8], initializer = tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable("W2",[10,1,8,8], initializer = tf.contrib.layers.xavier_initializer())

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters

def forward_propagation(X, parameters):
    '''propogate X forward through the network of two convolutional steps and a fully connected layer'''
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    # CONV2D:
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1], padding='VALID')
    # RELU
    A1 = tf.nn.relu(Z1)
    # CONV2D:
    Z2 = tf.nn.conv2d(A1,W2,strides=[1,1,1,1], padding='VALID')
    # RELU
    A2 = tf.nn.relu(Z2)
    # FLATTEN
    P2 = tf.contrib.layers.flatten(A2)
    # FULLY-CONNECTED without non-linear activation function.
    # 18 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(P2,18,activation_fn=None)
    
    return Z3

def compute_cost(Z3, Y):
    '''compute the cost of forward propagation'''
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z3,labels=Y))
    return cost

def model(X_train, Y_train, X_test, Y_test, parametersInitial=0, learning_rate = 0.0001,
          num_epochs = 100, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> CONV2D -> RELU -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 1000, 4, 1)
    Y_train -- test set, of shape (None, n_y = 18)
    X_test -- training set, of shape (None, 1000, 4, 1)
    Y_test -- test set, of shape (None, n_y = 18)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 5 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)

    # Initialize parameters
    parameters = initialize_parameters(parametersInitial)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3,Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            
            _ , tmp_cost = sess.run([optimizer,cost],feed_dict={X:X_train, Y:Y_train})
                
            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                # Calculate the correct predictions
                predict_op = tf.round(tf.sigmoid(Z3))
                correct_prediction = tf.equal(predict_op, Y)
                
                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
                test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
                
                print ("Cost after epoch %i: %f" % (epoch, tmp_cost), "Train Accuracy:", train_accuracy, "Test Accuracy:", test_accuracy)
                
            if print_cost == True and epoch % 1 == 0:
                costs.append(tmp_cost)
                
        return train_accuracy, test_accuracy, parameters


###########################################################################################
###################  Execute the model and Train data #####################################
###########################################################################################

#partition training/test data by 90/10 ratio
numTrain = int(numpts*.9)

X_train = data[0:numTrain]
Y_train = labels[0:numTrain,:]

X_test = data[numTrain:]
Y_test = labels[numTrain:,:]

#call the model
(train_accuracy, test_accuracy, parameters) = model(X_train, Y_train, X_test, Y_test)


