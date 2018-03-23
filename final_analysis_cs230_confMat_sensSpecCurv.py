import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import gzip as gz
import pickle


numpts = 10000 #total number of points in file is 767928, but can be reduced for now, for training


####################################################################################
####################    UPLOAD DATA AND LABELS FROM FILE   #########################
####################################################################################

labels = np.zeros((numpts,18))
data = np.zeros((numpts,1000,4,1))
goodGenes = []

verbose = True
        
with open('data/chr1_bp.txt','r') as f:
    genDict = {'n':4,'a':0, 'c':1,'t':2, 'g':3}
    badGenes = []
    geneCount = 0
    i = 0
    
    while (i < numpts):
        file_content = f.readline()
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
    file_content = str(file_content[0:-3])
    print(file_content)
    file_content = file_content.split('\\t')
    file_content[0] = file_content[0][2:-1]
    file_content[-1] = file_content[-1][0:-2]
    i=0
    geneCount = 0
    while i< numpts:
        file_content = f.readline()
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
    '''saves parameters to file as in main model code'''
    with open(file, 'wb') as f:
        pickle.dump(parameters, f)

def load_params(file):
    '''loads and returns parameters from file as in main model code'''
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


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    '''create placeholders for tensorflow X and Y'''
    X = tf.placeholder("float",shape=(None,n_H0,n_W0,n_C0))
    Y = tf.placeholder("float",shape=(None,n_y))
    
    return X, Y

def initialize_parameters(use_saved_params):
    '''create CNN filters.  If initP is present,
        use the given values, else use xavier initialization'''
    
    W1 = tf.get_variable("W1",[6,4,1,50])
    W2 = tf.get_variable("W2",[32,1,50,50])
    W3 = tf.get_variable("W3",[16,1,50,50])
    W4 = tf.get_variable("W4",[8,1,50,1])


    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4}
    
    return parameters

def forward_propagation(X, parameters, keep_prob_conv, keep_prob_fc):
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
    XX = tf.contrib.layers.fully_connected(XX,50,activation_fn=None)
    XX = tf.contrib.layers.batch_norm(XX)
    XX = tf.nn.sigmoid(XX)
    XX = tf.nn.dropout(XX,keep_prob_fc)
    Z3 = tf.contrib.layers.fully_connected(XX,18,activation_fn=None)
    
    return Z3

def compute_cost(Z3, Y):
    '''compute the cost of forward propagation'''
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z3,labels=Y))
    return cost

###########################################################################################
###################  Execute the model and Train data #####################################
###########################################################################################

#partition training/test data by 70/20/10 ratio
numTrain = int(numpts*.7)
numDev = int(numpts*.2)
numTest = numpts - numTrain - numDev

print("shuffling data with train/dev/dest ",numTrain,numDev,numTest)

(X_shuffle,Y_shuffle) = shuffle_data(data,labels)

X_train = X_shuffle[0:numTrain]
Y_train = Y_shuffle[0:numTrain]

X_dev = X_shuffle[numTrain:numTrain+numDev]
Y_dev = Y_shuffle[numTrain:numTrain+numDev]

#only test data will be used, but this is done for consistency with main model
X_test = X_shuffle[numTrain+numDev:]
Y_test = Y_shuffle[numTrain+numDev:]


#test the model
def test_model_multi_confusion_matrix(confInds, X_test = X_test, Y_test = Y_test):
    '''uses the test data to generate a confusion matrix from the given indices
        confInds -- array of ints e.g. [0,1,2] would generate the confusion matrix
                    for the first three cell types of the 18 cell types.
                    
        returns a confustion matrix of size 2**len(confInds)'''
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        
    (m, n_H0, n_W0, n_C0) = X_test.shape             
    n_y = Y_test.shape[1]
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)


    # Initialize saved parameters
    parameters = initialize_parameters(True)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    keep_prob_conv = 1
    keep_prob_fc = 1
    Z3 = forward_propagation(X, parameters, keep_prob_conv, keep_prob_fc)

    cost = compute_cost(Z3,Y)

    predict_op = tf.sigmoid(Z3)
    correct_prediction = tf.equal(tf.round(predict_op), Y)
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    auROC, roc_update_op = tf.metrics.auc(Y, predict_op, curve = 'ROC')
    auPRC, prc_update_op = tf.metrics.auc(Y, predict_op, curve = 'PR')
    print(Y.shape,predict_op.shape)


    # combine rows of the data X, so that a confusion matrix can be easily created by tensorflow

    if len(confInds)<2:
        print("you should pick more than one output index for this function")

    i = 0
    thresh = 0
    Ymulti = tf.reduce_mean(tf.slice(tf.cast(Y, dtype=tf.int32),[0,confInds[i]],[m,confInds[i]+1]),axis=1) #extract the first label
    predmulti = tf.reduce_mean(tf.slice(tf.cast(tf.round(predict_op)+thresh, dtype=tf.int32),[0,confInds[i]],[m,confInds[i]+1]),axis=1) #extract the first label
    while i<len(confInds)-1:
        i = i + 1
        Ymulti = Ymulti + (2**i) * tf.reduce_mean(tf.slice(tf.cast(Y, dtype=tf.int32),[0,confInds[i]],[m,confInds[i]+1]),axis=1)
        predmulti = predmulti + (2**i) * tf.reduce_mean(tf.slice(tf.cast(tf.round(predict_op)+thresh, dtype=tf.int32),[0,confInds[i]],[m,confInds[i]+1]),axis=1) #extract the first label

    #create confusion matrix
    conf_mat = tf.confusion_matrix(Ymulti, predmulti, num_classes = (2**len(confInds)))

    #spec_of_sens = tf.metrics.specificity_at_sensitivity(Y, predict_op, thresh)
    
    # Initialize all the variables globally
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        # Calculate the correct predictions
        saver.restore(sess, "best_metric_parameters3/model.ckpt")
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

        test_mat = conf_mat.eval({X: X_test, Y: Y_test})

        print(Ymulti.eval({X: X_test, Y: Y_test}))
        print(predmulti.eval({X: X_test, Y: Y_test}))
              
        
        return test_mat

#test the model
def test_model_average_confusion_matrix(X_test = X_test, Y_test = Y_test):
    ''' generate a confusion matrix for each of the 18 cell types
        and average them all to get one 2 by 2 confusion matrix

        returns a 2 by 2 array of the confusion matrix'''
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        
    (m, n_H0, n_W0, n_C0) = X_test.shape             
    n_y = Y_test.shape[1]
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)


    # Initialize saved parameters
    parameters = initialize_parameters(True)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    keep_prob_conv = 1
    keep_prob_fc = 1
    Z3 = forward_propagation(X, parameters, keep_prob_conv, keep_prob_fc)

    cost = compute_cost(Z3,Y)

    predict_op = tf.sigmoid(Z3)
    correct_prediction = tf.equal(tf.round(predict_op), Y)
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    auROC, roc_update_op = tf.metrics.auc(Y, predict_op, curve = 'ROC')
    auPRC, prc_update_op = tf.metrics.auc(Y, predict_op, curve = 'PR')
    print(Y.shape,predict_op.shape)
    
    conf_mats = []
    for conf_mat_ind in range(18):
        Yslice = tf.slice(tf.cast(Y, dtype=tf.int32),[0,0],[m,1])
        Yslice = tf.reduce_mean(Yslice, axis = 1)
        predslice = tf.slice(tf.cast(tf.round(predict_op),dtype=tf.int32),[0,0],[m,1])
        predslice = tf.reduce_mean(predslice, axis = 1)
        conf_mat = tf.confusion_matrix(Yslice, predslice, num_classes = 2)
        conf_mats.append(conf_mat)

    #spec_of_sens = tf.metrics.specificity_at_sensitivity(Y, predict_op, thresh)
    
    # Initialize all the variables globally
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        # Calculate the correct predictions
        saver.restore(sess, "best_metric_parameters3/model.ckpt")
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

        avg_mat = np.zeros((2,2))
        for i in range(18):
            print("doing conf matrix ",i)
            avg_mat += conf_mats[i].eval({X: X_test, Y: Y_test})

        avg_mat = avg_mat*(1/(1000*18))
        
        return avg_mat


#test the model
def test_model_sens_spec_curve(X_test = X_test, Y_test = Y_test):
    ''' generates a sensitivity and specificity curve from the test data

        returns sensitivity_values, specificity_values
                each between 0 and 1 that can be plotted'''
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        
    (m, n_H0, n_W0, n_C0) = X_test.shape             
    n_y = Y_test.shape[1]
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)


    # Initialize saved parameters
    parameters = initialize_parameters(True)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    keep_prob_conv = 1
    keep_prob_fc = 1
    Z3 = forward_propagation(X, parameters, keep_prob_conv, keep_prob_fc)

    cost = compute_cost(Z3,Y)

    predict_op = tf.sigmoid(Z3)
    correct_prediction = tf.equal(tf.round(predict_op), Y)
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    auROC, roc_update_op = tf.metrics.auc(Y, predict_op, curve = 'ROC')
    auPRC, prc_update_op = tf.metrics.auc(Y, predict_op, curve = 'PR')
    print(Y.shape,predict_op.shape)

        
    specs = []
    sens = np.linspace(0,1,101)
    for i in range(101):
        sen = sens[i]
        specs.append(tf.metrics.specificity_at_sensitivity(Y, predict_op, sen))
        
    #spec_of_sens = tf.metrics.specificity_at_sensitivity(Y, predict_op, thresh)
    
    # Initialize all the variables globally
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        # Calculate the correct predictions
        saver.restore(sess, "best_metric_parameters3/model.ckpt")
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
        
        
        specificity_values = np.zeros(101)
        sensitivity_values = np.linspace(0,1,101)
        for i in range(101):
            print(i)
            spec = specs[i][1].eval({X: X_test, Y: Y_test})
            specificity_values[i] = spec
        return sensitivity_values, specificity_values

#print(test_model_multi_confusion_matrix())
