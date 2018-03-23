import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import pickle


def load_params(file):
    with open(file, 'rb') as f:
        parameters = pickle.load(f)
    return parameters

def createGeneSeq(n):
    assert(n<4**6)
    listNum = list(np.base_repr(n, base = 4))
    listNum = np.array([int(num) for num in listNum])
    while len(listNum)<6:
            listNum = np.pad(listNum, (1,0), 'constant', constant_values=(0,0))
    geneSeq = np.zeros((6,4))
    for i in range(6):
        geneSeq[i,listNum[i]] = 1
    return geneSeq

def createGeneSeqLarge(n):
    assert(n<4**6)
    listNum = list(np.base_repr(n, base = 4))
    listNum = np.array([int(num) for num in listNum])
    while len(listNum)<6:
            listNum = np.pad(listNum, (1,0), 'constant', constant_values=(0,0))
    geneSeq = np.zeros((6,4,50))
    for i in range(6):
        geneSeq[i,listNum[i]] = np.ones(50)
    return geneSeq

def ind2gene(ind):
    basedict = {0:'a',1:'c',2:'t',3:'g'}
    gene = createGeneSeq(ind)
    basestr = ''
    for g in range(6):
        base = np.argmax(gene[g,:])
        basestr += basedict[base]
    return basestr

parameters = load_params('best_metric_params3.pkl')

W1 = parameters["W1"]
best = 0
maxInd = 0
for i in range(4**6):
    gene = createGeneSeqLarge(i)
    conv = np.multiply(W1[:,:,0,:],gene)
    mask = (conv>0)
    a = np.multiply(conv,mask)
    if np.sum(a)>best:
        best = np.sum(a)
        maxInd = i

print("best gene ",ind2gene(maxInd))

print("first filter size is ",W1.shape)

for filterNumb in range(50):
    maxInd = -1
    best = 0
    for i in range(4**6):
        #if np.mod(i,4**6//10)==0:
            #print("percent done ", (float(i)/(4**6)))
        gene = createGeneSeq(i)
        conv = np.multiply(W1[:,:,0,filterNumb],gene)
        mask = (conv>0)
        a = np.multiply(conv,mask)
        if np.sum(a)>best: # 0.000735 gives exactly the top 10
            #best = np.var(a)
            best = np.sum(a)
            maxInd = i

    basedict = {0:'a',1:'c',2:'t',3:'g'}

    
    print(ind2gene(maxInd))


print("here are some random ones")

for i in range(50):
    print(ind2gene(np.random.randint(0,4**6)))
         
##maxVals = []
##maxInds = []
##best = 0
##a_s = []
##for i in range(4**6):
##    #if np.mod(i,4**6//10)==0:
##        #print("percent done ", (float(i)/(4**6)))
##    gene = createGeneSeqLarge(i)
##    conv = np.multiply(W1[:,:,0,:],gene)
##    mask = (conv>0)
##    a = np.sum(np.multiply(conv,mask))
##    if a>0: # 0.000735 gives exactly the top 10
##        #best = np.var(a)
##        maxVals.append(a)
##        maxInds.append(i)
##        a_s.append(a)
##
##indSort = np.argsort(maxVals)
##maxValsSort = [maxVals[i] for i in indSort]
##maxIndsSort = [maxInds[i] for i in indSort]
##maxInds10 = maxIndsSort[-10:]
##print("top 10 indicies are ", maxInds10)
###with .73 set as thresh, aatgtt actually pairs with SOX2 which is use ot great blood stem cells
##
##basedict = {0:'a',1:'c',2:'t',3:'g'}
##
##for ind in maxInds10:
##    gene = createGeneSeq(ind)
##    basestr = ''
##    for g in range(6):
##        base = np.argmax(gene[g,:])
##        basestr += basedict[base]
##    print(basestr)
##    
