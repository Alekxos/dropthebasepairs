import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import gzip as gz
import pickle


numpts = 10000 #total number of points in file is 767928, but can be reduced for now, for training

#upload data and labels from files
labels = np.zeros((numpts,18))
data = np.zeros((numpts,1000,4,1))
goodGenes = []

skip = 0
skip2 = skip

verbose = True
        
with open('data/chr1_bp.txt','r') as f:
    genDict = {'n':4,'a':0, 'c':1,'t':2, 'g':3}
    badGenes = []
    geneCount = 0
    i = 0
    
    while (i < numpts):
        
        file_content = f.readline()
        if np.mod(i,100)==0 : print(i, file_content)
        file_content = f.readline()
        #if i<10: print(i, file_content)
        while skip>0:
            skip -= 1
            file_content = f.readline()
            
        geneseq = list(file_content)
        geneseq = geneseq[0:-1]
        if np.mod(i,int(numpts/10))==0:
            print(geneseq[0:10])
        basecount = 0

        #create single gene
        dataSingleGene = data[i,:,:,0]
        #print(dataSingleGene.shape)
        skipGene = False
        for base in geneseq:
            genInd = genDict[base.lower()]
            if genInd!=4:
                dataSingleGene[basecount,genInd] = 1
            else:
                skipGene = True
                #print(i)
            basecount += 1

        #add to data if its a good gene
        if not skipGene:
            
            data[i,:,:,0] = dataSingleGene
            i += 1
        else:
            print(geneCount)
            badGenes.append(geneCount)
                
        geneCount += 1

        
with gz.open('hema_model_jason_only_oct17_idr_K562_GM12878_CV_corePeakSize_500_totalPeakSize_1000_neg_flank_offset_500_Sequential_1D_model_binaryPeaks.txt.gz', 'rb') as f:
    file_content = f.readline()
    
    while skip2>0:
        skip2 -= 1
        file_content = f.readline()
    i=0
    geneCount = 0
    while i< numpts:
        file_content = f.readline()
        if np.mod(i,100)==0 : print(i, file_content)
        file_content = str(file_content[0:-1])
        file_content = file_content.split('\\t')
        file_content[0] = file_content[0][2:-1]
        file_content[-1] = file_content[-1][0:-1]
        
        if geneCount not in badGenes:
            labels[i][:] = [int(cont) for cont in file_content[4:]]
            if np.mod(i,int(numpts/10))==0 and verbose:
                print(i,len(file_content[4:]),file_content[0], labels[i])
            i += 1
            
        geneCount += 1   
        

print("there are ",len(badGenes)," bad genes with 'N' in them")
####################################################################################
####################    BEGIN MODEL   ##############################################
####################################################################################
#code taken largely from Coursera class 4 on CNN's in tensorflow

def save_params(parameters, file):
    with open(file, 'wb') as f:
        pickle.dump(parameters, f)

def load_params(file):
    with open(file, 'rb') as f:
        parameters = pickle.load(f)
    return parameters

def shuffle_data(X,Y):
    '''first index of both X and Y will be shuffled in unison'''
    shuffleXY = [(X[i],Y[i]) for i in range(X.shape[0])]
    np.random.seed(1)
    np.random.shuffle(shuffleXY)
    X_shuffle = np.zeros(X.shape)
    Y_shuffle = np.zeros(Y.shape)
    for i in range(X.shape[0]):
        X_shuffle[i] = shuffleXY[i][0]
        Y_shuffle[i] = shuffleXY[i][1]
    return(X_shuffle,Y_shuffle)

def create_minibatches(X_train, Y_train):
    '''creates minibatches of size 32
        X_batches size is (numMinibatches, X_train.shape)'''
    
    numSamples = X_train.shape[0]
    numMinibatches = numSamples//32
    X_batches = []
    Y_batches = []
    for i in range(numMinibatches):
        index = i*32
        X_batches.append(X_train[index:index+32])
        Y_batches.append(Y_train[index:index+32])
        
    if numMinibatches*32 < numSamples:
        X_batches.append(X_train[index+33:])
        Y_batches.append(Y_train[index+33:])

    return (X_batches,Y_batches)

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    '''create placeholders for tensorflow X and Y'''
    X = tf.placeholder("float",shape=(None,n_H0,n_W0,n_C0))
    Y = tf.placeholder("float",shape=(None,n_y))
    
    return X, Y

def initialize_parameters(use_saved_params, beta_reg):
    '''create CNN filters.  If initP is present,
        use the given values, else use xavier initialization'''
    if not use_saved_params:
##        initP = load_params('bestmetric_params.pkl')
##        W1 = tf.get_variable("W1", initializer=tf.constant(initP["W1"]))
##        W2 = tf.get_variable("W2", initializer=tf.constant(initP["W2"]))
##        W3 = tf.get_variable("W3", initializer=tf.constant(initP["W3"]))
##        W4 = tf.get_variable("W4", initializer=tf.constant(initP["W4"]))
##        W5 = tf.get_variable("W5", initializer=tf.constant(initP["W5"]))
##        W6 = tf.get_variable("W6", initializer=tf.constant(initP["W6"]))
        W1 = tf.get_variable("W1",[6,4,1,50], initializer = tf.contrib.layers.xavier_initializer(),
                             regularizer=tf.contrib.layers.l2_regularizer(tf.constant(beta_reg, dtype=tf.float32)))
        W2 = tf.get_variable("W2",[32,1,50,50], initializer = tf.contrib.layers.xavier_initializer(),
                             regularizer=tf.contrib.layers.l2_regularizer(tf.constant(beta_reg, dtype=tf.float32)))
        W3 = tf.get_variable("W3",[16,1,50,50], initializer = tf.contrib.layers.xavier_initializer(),
                             regularizer=tf.contrib.layers.l2_regularizer(tf.constant(beta_reg, dtype=tf.float32)))
        W4 = tf.get_variable("W4",[8,1,50,1], initializer = tf.contrib.layers.xavier_initializer(),
                             regularizer=tf.contrib.layers.l2_regularizer(tf.constant(beta_reg, dtype=tf.float32)))
        
        
    else:        
##        initP = load_params('best_params1.pkl')
##        
##        W1 = tf.get_variable("W1", initializer=tf.constant(initP["W1"]))
##        W2 = tf.get_variable("W2", initializer=tf.constant(initP["W2"]))
##        W3 = tf.get_variable("W3", initializer=tf.constant(initP["W3"]))
##        W4 = tf.get_variable("W4", initializer=tf.constant(initP["W4"]))

        
        W1 = tf.get_variable("W1",[6,4,1,50],
                             regularizer=tf.contrib.layers.l2_regularizer(tf.constant(beta_reg, dtype=tf.float32)))
        W2 = tf.get_variable("W2",[32,1,50,50],
                             regularizer=tf.contrib.layers.l2_regularizer(tf.constant(beta_reg, dtype=tf.float32)))
        W3 = tf.get_variable("W3",[16,1,50,50],
                             regularizer=tf.contrib.layers.l2_regularizer(tf.constant(beta_reg, dtype=tf.float32)))
        W4 = tf.get_variable("W4",[8,1,50,1],
                             regularizer=tf.contrib.layers.l2_regularizer(tf.constant(beta_reg, dtype=tf.float32)))


    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4}
    
    return parameters

def forward_propagation(X, parameters, keep_prob_conv, keep_prob_fc, beta_reg):
    '''propogate X forward through the network of two convolutional steps and a fully connected layer'''
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']

    # CONV2D:
    XX = tf.nn.conv2d(X,W1,strides=[1,1,1,1], padding='VALID')
    XX = tf.contrib.layers.batch_norm(XX)
    XX = tf.nn.relu(XX)
    XX = tf.nn.max_pool(XX, [1,4,1,1],[1,4,1,1], 'VALID')
    # CONV2D:
    XX = tf.nn.dropout(XX,keep_prob_conv)
    XX = tf.nn.conv2d(XX,W2,strides=[1,1,1,1], padding='VALID')
    XX = tf.contrib.layers.batch_norm(XX)
    XX = tf.nn.relu(XX)
    XX = tf.nn.max_pool(XX, [1,4,1,1],[1,4,1,1], 'VALID')
    # CONV2D:
    XX = tf.nn.dropout(XX,keep_prob_conv)
    XX = tf.nn.conv2d(XX,W3,strides=[1,1,1,1], padding='VALID')
    XX = tf.contrib.layers.batch_norm(XX)
    XX = tf.nn.relu(XX)
    # CONV1D: Bottleneck
    XX = tf.nn.dropout(XX,keep_prob_conv)
    XX = tf.nn.conv2d(XX,W4,strides=[1,1,1,1], padding='VALID')
    XX = tf.contrib.layers.batch_norm(XX)
    XX = tf.nn.relu(XX)
    # FLATTEN
    XX = tf.contrib.layers.flatten(XX)
    # FULLY-CONNECTED layers with non-linear activation functions.
    # 18 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    XX = tf.nn.dropout(XX,keep_prob_fc)
    XX = tf.contrib.layers.fully_connected(XX,50,activation_fn=None,
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(tf.constant(beta_reg, dtype=tf.float32)))
    XX = tf.contrib.layers.batch_norm(XX)
    XX = tf.nn.sigmoid(XX)
    XX = tf.nn.dropout(XX,keep_prob_fc)
    Z3 = tf.contrib.layers.fully_connected(XX,18,activation_fn=None,
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(tf.constant(beta_reg, dtype=tf.float32)))
    
    return Z3

def compute_cost(Z3, Y):
    '''compute the cost of forward propagation'''
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z3,labels=Y))
    cost += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
    return cost

def model(X_train, Y_train, X_test, Y_test, use_saved_params = False, use_metric = False, learning_rate = 0.00005,
          num_epochs = 10, keep_prob_conv = 1, keep_prob_fc = 1, beta_reg = .01):
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
    print_cost -- True to print the cost every 5 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    
    X_minibatches,Y_minibatches = create_minibatches(X_train,Y_train)
        
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)

    # Initialize parameters
    parameters = initialize_parameters(use_saved_params, beta_reg)

    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters, keep_prob_conv, keep_prob_fc, beta_reg)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3,Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    pred_thresh = .5
    predict_op = tf.round(tf.sigmoid(Z3)-.5+pred_thresh)
    correct_prediction = tf.equal(predict_op, Y)
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    auROC, roc_update_op = tf.metrics.auc(Y, predict_op, curve = 'ROC')
    auPRC, prc_update_op = tf.metrics.auc(Y, predict_op, curve = 'PR')
    
    # Initialize all the variables globally
    #init = tf.global_variables_initializer()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)

        if use_saved_params and use_metric:
            saver.restore(sess, "best_metric_parameters3/model.ckpt")
            print("Model restored.")
            test_auROC, test_auPRC = sess.run([auROC, auPRC], feed_dict = {X: X_test, Y: Y_test})
            best_metric = test_auPRC #per Peyton's recommendation
            best_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        elif use_saved_params:
            saver.restore(sess, "best_accuracy_parameters3/model.ckpt")
            print("Model restored.")
            test_auROC, test_auPRC = sess.run([auROC, auPRC], feed_dict = {X: X_test, Y: Y_test})
            best_metric = test_auPRC #per Peyton's recommendation
            best_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        else:
            best_metric = 0
            best_accuracy = 0

        # Do the training loop
        for epoch in range(1,num_epochs+1):

            epoch_cost = 0
            for i in range(len(X_minibatches)):
                _ , batch_cost = sess.run([optimizer,cost],feed_dict={X:X_minibatches[i], Y:Y_minibatches[i]})
                epoch_cost = epoch_cost + (32/len(X_train))*batch_cost
            costs.append(epoch_cost)    

            #evaluate performance of epoch
            train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
            test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
            sess.run([roc_update_op,prc_update_op], feed_dict = {X: X_test, Y: Y_test})
            test_auROC, test_auPRC = sess.run([auROC, auPRC], feed_dict = {X: X_test, Y: Y_test})
            test_metric = test_auPRC #per Peyton's recommendation
            print ("Epoch cost %i: %f" % (epoch, epoch_cost),
                   "Train Accuracy:", train_accuracy,
                   "Dev Accuracy:", test_accuracy,
                   "auROC:", test_auROC,
                   "auPRC:", test_auPRC,
                   "metric", test_metric)
            
            if test_metric > best_metric:
                save_params(sess.run(parameters),'best_metric_params3.pkl')
                best_metric = test_metric
                # Save the variables to disk.
                save_path = saver.save(sess, "best_metric_parameters3/model.ckpt")
                print("Model saved in path: %s" % save_path)

            if test_accuracy > best_accuracy:
                save_params(sess.run(parameters),'best_accuracy_params3.pkl')
                best_accuracy = test_accuracy
                # Save the variables to disk.
                save_path = saver.save(sess, "best_accuracy_parameters3/model.ckpt")
                print("Model saved in path: %s" % save_path)

        parameters = sess.run(parameters)
        return best_metric, best_accuracy, parameters



###########################################################################################
###################  Execute the model and Train data #####################################
###########################################################################################

##numpts = len(goodGenes)
##data = data[goodGenes]
##labels = labels[goodGenes]

#partition training/test data by 80/10/10 ratio

print("shuffling ",numpts," data points")

numTrain = int(numpts*.7)
numDev = int(numpts*.2)
numTest = numpts - numTrain - numDev

augmentData = True
#print("train/dev/test break down ", numTrain, numDev, numTest)

(X_shuffle,Y_shuffle) = shuffle_data(data,labels)

#allocate training and dev set with augmentation
X_train = X_shuffle[0:numTrain]
Y_train = Y_shuffle[0:numTrain]

if augmentData:
    mask = (np.sum(Y_train,axis=1)>0)
    X_train = X_train[mask]
    Y_train = Y_train[mask]

X_dev = X_shuffle[numTrain:numTrain+numDev]
Y_dev = Y_shuffle[numTrain:numTrain+numDev]

if augmentData:
    mask = (np.sum(Y_dev,axis=1)>0)
    X_dev = X_dev[mask]
    Y_dev = Y_dev[mask]

#allocate test set
X_test = X_shuffle[numTrain+numDev:]
Y_test = Y_shuffle[numTrain+numDev:]

print("data has size ",X_train.shape,X_dev.shape,X_test.shape)

#test the model
def test_model(X_test = X_test, Y_test = Y_test):
    """
    Returns:
    test_accuracy, test_auROC, test_auPRC, test_metric
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        
    (m, n_H0, n_W0, n_C0) = X_test.shape             
    n_y = Y_test.shape[1]

    keep_prob_conv = 1
    keep_prob_fc = 1
    beta_reg = 0
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)

    # Initialize saved parameters
    try:
        parameters = initialize_parameters(True, beta_reg)
    except:
        print("you must first run the model and save parameters.  I'd recommend setting best_metric = 0, so that you make sure to save parameters")

    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    
    Z3 = forward_propagation(X, parameters, keep_prob_conv, keep_prob_fc, beta_reg)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3,Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predict_op = (tf.sigmoid(Z3))
    correct_prediction = tf.equal(tf.round(predict_op), Y)
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    auROC, roc_update_op = tf.metrics.auc(Y, predict_op, curve = 'ROC')
    auPRC, prc_update_op = tf.metrics.auc(Y, predict_op, curve = 'PR')
    
    # Initialize all the variables globally
    #init = tf.global_variables_initializer()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        # Calculate the correct predictions
        saver.restore(sess, "best_accuracy_parameters3/model.ckpt")
        print("Model restored.")

        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        sess.run([roc_update_op,prc_update_op], feed_dict = {X: X_test, Y: Y_test})
        test_auROC, test_auPRC = sess.run([auROC, auPRC], feed_dict = {X: X_test, Y: Y_test})
        #test_metric = (1/2)*(test_auROC + test_auPRC)
        test_metric = test_auPRC #per Peyton's recommendation
        print ("Test Accuracy:", test_accuracy,
               "auROC:", test_auROC,
               "auPRC:", test_auPRC,
               "test_metric", test_metric)
        #save_path = saver.save(sess, "best_model_parameters/model.ckpt")
        #print("Model saved in path: %s" % save_path)
        
        return test_accuracy, test_auROC, test_auPRC, test_metric


#dropout tries: tried to dropout everything at .9 but it still overfit after 85%.  
#   to get up accuracy I tried saving the best parameters from each short run
#   and going back and forth between dropout of .9 and dropout of 1.
#   this seemed to help the model because it was overfitting a lot before.
#choosing keep_prob_fc = .9 and keep_prob_conv = 1 seems to work very well
#over 90% accuracy its best to choose a learning_rate=.0005
#switching over to optimize ROC and PRC now.  I chose a metric of auroc+auprc
#good starter code: (best_test_metric, parameters) = model(X_train, Y_train, X_test, Y_test, load_params('bestmetric_params.pkl'), learning_rate = .0005, num_epochs = 5, best_metric = .65, keep_prob_fc = .8, keep_prob_conv = .9)
#good starter code: (best_test_metric, parameters) = model(X_train, Y_train, X_dev, Y_dev, use_saved_params=True, learning_rate = .0005, num_epochs = 5, best_metric = .65, keep_prob_fc = .9, keep_prob_conv = .9)
#good starter code: best_metric, best_accuracy, parameters = model(X_train, Y_train, X_dev, Y_dev, use_saved_params = True, learning_rate = 0.001, num_epochs = 10, keep_prob_conv = .9, keep_prob_fc = .9, beta_reg = 0)

#best_metric, best_accuracy, parameters = model(X_train, Y_train, X_dev, Y_dev, use_saved_params = False, use_metric = False, learning_rate = 0.005, num_epochs = 10, keep_prob_conv = .9, keep_prob_fc = .9, beta_reg = .001)
