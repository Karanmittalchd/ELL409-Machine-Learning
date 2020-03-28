#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import math
from numpy import asarray
from numpy import savetxt
import matplotlib.pyplot as plt 
def softmax (A):
    B = np.exp(A)
    for i in range(len(B)):
        if(B[i] < 0.000000000000000000000000001):
            B[i] = 0
    B = B/np.sum(B)
    return B;


def sigmoid(A):
    return 1/(1+np.exp(-A))
def sigmoidd(A):
    return sigmoid(A)*(1-sigmoid(A))
def tanh(A):
    return ((np.exp(A)-np.exp(-A))/(np.exp(A)+np.exp(-A)))
def relu(A):
    for i in range(len(A)):
        if(A[i]<=0):
            A[i]=0
    return A
def relud(A):
    for i in range(len(A)):
        if(A[i]==0):
            A[i]=0
        elif(A[i]>0):
            A[i]=1
    return A
            


# In[2]:


train_data = np.loadtxt("mnist_train.csv", delimiter=',')
labels = np.array(train_data[:,0])
a_s=int(0.8*len(train_data))
labels_train = labels[:a_s]
labels_test = labels[a_s:]
train_data = train_data / 1000
batchsize = 1
bs = batchsize
n = int(a_s/bs)
neurons = 75
alpha = 0.01
epochs =60
#Loss Function used is Log Loss
#Using Activation Function Sigmoid
b1 = 0.00001*np.random.randn(neurons,1)
b2 = 0.00001*np.random.randn(10,1)
W1 = 0.00001*np.random.randn(neurons,784)
W2 = 0.00001*np.random.randn(10,neurons)
test_data = np.loadtxt("C:\Sem VI\ELL409\Assignments\mnist_test.csv", delimiter=',')

for k in range (epochs):
    abc=0
    batchsize = bs
    for ij in range(n):
        if(ij==n-1):
            batchsize = a_s
        sz = batchsize-abc

        delta = 0
        for i in range(abc,batchsize):    
            train_pattern = np.transpose(train_data[i:i+1,1:])
            label = int(labels[i])
            hidden_l = sigmoid(np.dot(W1,train_pattern)+b1)

            output_l = softmax(np.dot(W2,hidden_l)+b2)
            d_output = np.zeros([10,1],dtype=float)
            d_output[label]=1
            delta = delta + d_output-output_l
            deltaT = delta.T
        hiddenT = hidden_l.T
        dl = np.dot(delta/sz,hiddenT)
        one = np.ones(hidden_l.shape)
        change = alpha*dl
        W2 = W2+change
        b2 = b2+alpha*(delta/sz)
        sigmo_d = np.multiply(hidden_l,(one-hidden_l))
        s2 = np.dot(W2.T,delta/sz)
        s3 = np.transpose(np.multiply(sigmo_d,s2))
        b1 = b1+alpha*(s3.T)
        dl2 = np.dot(train_pattern,s3)
        W1 = W1+alpha*(dl2.T)
        abc = abc+bs
        batchsize = batchsize+bs

#Accuracy Calculation        
q=0
for a in range(a_s,len(train_data)):
    pattern = np.transpose(train_data[a:a+1,1:])
    o_label = labels_test[a-a_s]
    hid = sigmoid(np.dot(W1,pattern)+b1)
    output = softmax(np.dot(W2,hid)+b2)
    oput = np.argmax(output)

    if o_label==oput:
        q+=1
accuracy = 100*q/(len(train_data)-a_s)
print(accuracy)


# In[5]:


k=0
ij=0
i=0
a=0
neurons = 75
alpha = 0.01
epochs =60
batchsize = 1
bs = batchsize
n = int(a_s/bs)
#Using activation function Tanh
b1 = 0.01*np.random.randn(neurons,1)
b2 = 0.01*np.random.randn(10,1)
W1 = 0.01*np.random.randn(neurons,784)
W2 = 0.01*np.random.randn(10,neurons)
test_data = np.loadtxt("C:\Sem VI\ELL409\Assignments\mnist_test.csv", delimiter=',')

for k in range(epochs):
    abc=0
    batchsize = bs
    for ij in range(n):
        if(ij==n-1):
            batchsize = a_s
        sz = batchsize-abc
        
        delta = 0
        for i in range(abc,batchsize):    
            train_pattern = np.transpose(train_data[i:i+1,1:])
            label = int(labels[i])
            hidden_l = tanh(np.dot(W1,train_pattern)+b1)

            output_l = softmax(np.dot(W2,hidden_l)+b2)
            d_output = np.zeros([10,1],dtype=float)
            d_output[label]=1
            delta = delta + d_output-output_l
            deltaT = delta.T
        hiddenT = hidden_l.T
        dl = np.dot(delta/sz,hiddenT)
        one = np.ones(hidden_l.shape)
        alpha = 0.01
        change = alpha*dl
        W2 = W2+change
        b2 = b2+alpha*(delta/sz)
        sigmo_d = one - np.multiply(hidden_l,hidden_l)
        s2 = np.dot(W2.T,delta/sz)
        s3 = np.transpose(np.multiply(sigmo_d,s2))
        b1 = b1+alpha*(s3.T)
        dl2 = np.dot(train_pattern,s3)
        W1 = W1+alpha*(dl2.T)
        abc = abc+bs
        batchsize = batchsize+bs

q=0
for a in range(a_s,len(train_data)):
    pattern = np.transpose(train_data[a:a+1,1:])
    o_label = labels_test[a-a_s]
    hid = tanh(np.dot(W1,pattern)+b1)
    output = softmax(np.dot(W2,hid)+b2)
    oput = np.argmax(output)

    if o_label==oput:
        q+=1
accuracy = 100*q/(len(train_data)-a_s)
print(accuracy) 


# In[6]:


k=0
ij=0
i=0
a=0
neurons = 75
alpha = 0.01
epochs =60
batchsize = 1
bs = batchsize
n = int(a_s/bs)
#Using Activation Function Relu
b1 = 0.00001*np.random.randn(neurons,1)
b2 = 0.00001*np.random.randn(10,1)
W1 = 0.00001*np.random.randn(neurons,784)
W2 = 0.00001*np.random.randn(10,neurons)
test_data = np.loadtxt("C:\Sem VI\ELL409\Assignments\mnist_test.csv", delimiter=',')

for k in range (epochs):
    abc=0
    batchsize = bs
    for ij in range(n):
        if(ij==n-1):
            batchsize = a_s
        sz = batchsize-abc

        delta = 0
        for i in range(abc,batchsize):    
            train_pattern = np.transpose(train_data[i:i+1,1:])
            label = int(labels[i])
            hidden_l = sigmoid(np.dot(W1,train_pattern)+b1)

            output_l = softmax(np.dot(W2,hidden_l)+b2)
            d_output = np.zeros([10,1],dtype=float)
            d_output[label]=1
            delta = delta + d_output-output_l
            deltaT = delta.T
        hiddenT = hidden_l.T
        dl = np.dot(delta/sz,hiddenT)
        one = np.ones(hidden_l.shape)
        change = alpha*dl
        W2 = W2+change
        b2 = b2+alpha*(delta/sz)
        sigmo_d = np.multiply(hidden_l,(one-hidden_l))
        s2 = np.dot(W2.T,delta/sz)
        s3 = np.transpose(np.multiply(sigmo_d,s2))
        b1 = b1+alpha*(s3.T)
        dl2 = np.dot(train_pattern,s3)
        W1 = W1+alpha*(dl2.T)
        abc = abc+bs
        batchsize = batchsize+bs

#Accuracy Calculation        
q=0
for a in range(a_s,len(train_data)):
    pattern = np.transpose(train_data[a:a+1,1:])
    o_label = labels_test[a-a_s]
    hid = sigmoid(np.dot(W1,pattern)+b1)
    output = softmax(np.dot(W2,hid)+b2)
    oput = np.argmax(output)

    if o_label==oput:
        q+=1
accuracy = 100*q/(len(train_data)-a_s)
print(accuracy)


# In[24]:


#Variation with the number of layers and neurons 
hidden_layers = 2
neurons = []
m=0
k=0
i=0

for m in range(hidden_layers):
    n = input()
    neurons.append(int(n))

W1 = 0.1*np.random.randn(neurons[0],784)
WN = 0.1*np.random.randn(10,neurons[hidden_layers-1])
W = []
W.append(W1)
m = 0
for m in range(1,hidden_layers):
    W.append(0.01*np.random.randn(neurons[m],neurons[m-1]))
W.append(WN)
test_data = np.loadtxt("C:\Sem VI\ELL409\Assignments\mnist_test.csv", delimiter=',')

for k in range (1000):
    for i in range(a_s):    
        train_pattern = np.transpose(train_data[i:i+1,1:])
        label = int(labels_train[i])
        hidden_1 = sigmoid(np.dot(W[0],train_pattern))
        hidden_l = []
        hidden_l.append(hidden_1)
        for v in range(1,hidden_layers):
            hidden_l.append(sigmoid(np.dot(W[v],hidden_l[v-1])))
        output_l = softmax(np.dot(W[hidden_layers],hidden_l[hidden_layers-1]))
        d_output = np.zeros([10,1],dtype=float)
        d_output[label]=1
        delta = d_output-output_l
        alpha = 0.01
        dl = np.dot(delta,hidden_l[hidden_layers-1].T)
        W[hidden_layers]=W[hidden_layers]+alpha*dl
        for wt in range(hidden_layers-1):
            mid = np.dot(W[hidden_layers-wt].T,delta)
            delta = np.multiply(sigmoidd(hidden_l[hidden_layers-wt-1]), mid)
            dl1 = np.dot(hidden_l[hidden_layers-wt-2],delta.T)
            W[hidden_layers-wt-1] = W[hidden_layers-wt-1] + alpha*(dl1.T)
        mid = np.dot(W[1].T,delta)
        delta = np.multiply(sigmoidd(hidden_l[0]), mid)
        dl1 = np.dot(train_pattern,delta.T)
        W[0] = W[0] + alpha*(dl1.T)
q=0
for a in range(a_s,len(train_data)):
    pattern = np.transpose(train_data[a:a+1,1:])
    o_label = labels[a]
    hid = sigmoid(np.dot(W[0],pattern))
    h1 = []
    h1.append(hid)
    for cd in range(1,hidden_layers):
        h1.append(sigmoid(np.dot(W[cd],h1[cd-1])))
    output = softmax(np.dot(W[hidden_layers],h1[hidden_layers-1]))
    oput = np.argmax(output)
    if o_label==oput:
        q+=1
accuracy = 100*q/(len(train_data)-a_s)
print(accuracy)

   


# In[ ]:


k=0
ij=0
i=0
a=0
neurons = 75
alpha = 0.01
epochs =60
batchsize = 1
bs = batchsize
n = int(a_s/bs)
#Variation with Learning Rate
#alpha is the learning rate
alpha = 0.1
xval = []
yval = []
while(alpha>0.0000001):
    b1 = 0.00001*np.random.randn(neurons,1)
    b2 = 0.00001*np.random.randn(10,1)
    W1 = 0.00001*np.random.randn(neurons,784)
    W2 = 0.00001*np.random.randn(10,neurons)
    test_data = np.loadtxt("C:\Sem VI\ELL409\Assignments\mnist_test.csv", delimiter=',')

    for k in range (60):
        abc=0
        batchsize = bs
        for ij in range(n):
            if(ij==n-1):
                batchsize = a_s
            sz = batchsize-abc

            delta = 0
            for i in range(abc,batchsize):    
                train_pattern = np.transpose(train_data[i:i+1,1:])
                label = int(labels[i])
                hidden_l = sigmoid(np.dot(W1,train_pattern)+b1)
                output_l = softmax(np.dot(W2,hidden_l)+b2)
                d_output = np.zeros([10,1],dtype=float)
                d_output[label]=1
                delta = delta + d_output-output_l
                deltaT = delta.T
            hiddenT = hidden_l.T
            dl = np.dot(delta/sz,hiddenT)
            one = np.ones(hidden_l.shape)
            change = alpha*dl
            W2 = W2+change
            b2 = b2+alpha*(delta/sz)
            sigmo_d = np.multiply(hidden_l,(one-hidden_l))
            s2 = np.dot(W2.T,delta/sz)
            s3 = np.transpose(np.multiply(sigmo_d,s2))
            b1 = b1+alpha*(s3.T)
            dl2 = np.dot(train_pattern,s3)
            W1 = W1+alpha*(dl2.T)
            abc = abc+bs
            batchsize = batchsize+bs
    
    q=0
    for a in range(a_s,len(train_data)):
        pattern = np.transpose(train_data[a:a+1,1:])
        o_label = labels_test[a-a_s]
        hid = sigmoid(np.dot(W1,pattern)+b1)
        output = softmax(np.dot(W2,hid)+b2)
        oput = np.argmax(output)

        if o_label==oput:
            q+=1
    accuracy = 100*q/(len(train_data)-a_s)
    print(accuracy)
    xval.append(math.log(alpha))
    yval.append(accuracy)
    alpha = alpha/10
plt.plot(xval,yval)
 
   


# In[19]:


k=0
ij=0
i=0
a=0
neurons = 75
alpha = 0.01
epochs =60
batchsize = 1
bs = batchsize
n = int(a_s/bs)
#Variation with the number of epochs
alpha = 0.01
epochs = 10
xval = []
yval = []
while(epochs<=100):
    b1 = 0.00001*np.random.randn(neurons,1)
    b2 = 0.00001*np.random.randn(10,1)
    W1 = 0.00001*np.random.randn(neurons,784)
    W2 = 0.00001*np.random.randn(10,neurons)
    test_data = np.loadtxt("C:\Sem VI\ELL409\Assignments\mnist_test.csv", delimiter=',')

    for k in range (epochs):
        abc=0
        batchsize = bs
        for ij in range(n):
            if(ij==n-1):
                batchsize = a_s
            sz = batchsize-abc

            delta = 0
            for i in range(abc,batchsize):    
                train_pattern = np.transpose(train_data[i:i+1,1:])
                label = int(labels[i])
                hidden_l = sigmoid(np.dot(W1,train_pattern)+b1)

                output_l = softmax(np.dot(W2,hidden_l)+b2)
                d_output = np.zeros([10,1],dtype=float)
                d_output[label]=1
                delta = delta + d_output-output_l
                deltaT = delta.T
            hiddenT = hidden_l.T
            dl = np.dot(delta/sz,hiddenT)
            one = np.ones(hidden_l.shape)
            change = alpha*dl
            W2 = W2+change
            b2 = b2+alpha*(delta/sz)
            sigmo_d = np.multiply(hidden_l,(one-hidden_l))
            s2 = np.dot(W2.T,delta/sz)
            s3 = np.transpose(np.multiply(sigmo_d,s2))
            b1 = b1+alpha*(s3.T)
            dl2 = np.dot(train_pattern,s3)
            W1 = W1+alpha*(dl2.T)
            abc = abc+bs
            batchsize = batchsize+bs
    
    q=0
    for a in range(a_s,len(train_data)):
        pattern = np.transpose(train_data[a:a+1,1:])
        o_label = labels_test[a-a_s]
        hid = sigmoid(np.dot(W1,pattern)+b1)
        output = softmax(np.dot(W2,hid)+b2)
        oput = np.argmax(output)

        if o_label==oput:
            q+=1
    accuracy = 100*q/(len(train_data)-a_s)
    print(accuracy)
    xval.append(epochs)
    yval.append(accuracy)
    epochs = epochs+10
plt.plot(xval,yval)


# In[13]:


k=0
ij=0
i=0
a=0
batchsize = 1
bs = batchsize
n = int(a_s/bs)
neurons = 75
alpha = 0.01
epochs =60
#Variation with the number of neurons in a layer
neurons = 10
alpha = 0.01
epochs =60
xval = []
yval = []
while(neurons<=100):
    b1 = 0.00001*np.random.randn(neurons,1)
    b2 = 0.00001*np.random.randn(10,1)
    W1 = 0.00001*np.random.randn(neurons,784)
    W2 = 0.00001*np.random.randn(10,neurons)
    test_data = np.loadtxt("C:\Sem VI\ELL409\Assignments\mnist_test.csv", delimiter=',')

    for k in range (epochs):
        abc=0
        batchsize = bs
        for ij in range(n):
            if(ij==n-1):
                batchsize = a_s
            sz = batchsize-abc
            delta = 0
            for i in range(abc,batchsize):    
                train_pattern = np.transpose(train_data[i:i+1,1:])
                label = int(labels[i])
                hidden_l = sigmoid(np.dot(W1,train_pattern)+b1)
                output_l = softmax(np.dot(W2,hidden_l)+b2)
                d_output = np.zeros([10,1],dtype=float)
                d_output[label]=1
                delta = delta + d_output-output_l
                deltaT = delta.T
            hiddenT = hidden_l.T
            dl = np.dot(delta/sz,hiddenT)
            one = np.ones(hidden_l.shape)
            change = alpha*dl
            W2 = W2+change
            b2 = b2+alpha*(delta/sz)
            sigmo_d = np.multiply(hidden_l,(one-hidden_l))
            s2 = np.dot(W2.T,delta/sz)
            s3 = np.transpose(np.multiply(sigmo_d,s2))
            b1 = b1+alpha*(s3.T)
            dl2 = np.dot(train_pattern,s3)
            W1 = W1+alpha*(dl2.T)
            abc = abc+bs
            batchsize = batchsize+bs
    
    q=0
    for a in range(a_s,len(train_data)):
        pattern = np.transpose(train_data[a:a+1,1:])
        o_label = labels_test[a-a_s]
        hid = sigmoid(np.dot(W1,pattern)+b1)
        output = softmax(np.dot(W2,hid)+b2)
        oput = np.argmax(output)

        if o_label==oput:
            q+=1
    accuracy = 100*q/(len(train_data)-a_s)
    print(accuracy)
    xval.append(neurons)
    yval.append(accuracy)
    neurons = neurons+10
plt.plot(xval,yval)
  


# In[18]:


k=0
ij=0
i=0
a=0
neurons = 75
alpha = 0.01
epochs =60
batchsize = 1
bs = batchsize
n = int(a_s/bs)
#Variation with Batch Size
alpha = 0.01
epochs = 60
batchsize = 1
bs = batchsize
xval = []
yval = []
while(bs<=10):
    n = int(a_s/bs)
    b1 = 0.00001*np.random.randn(neurons,1)
    b2 = 0.00001*np.random.randn(10,1)
    W1 = 0.00001*np.random.randn(neurons,784)
    W2 = 0.00001*np.random.randn(10,neurons)
    test_data = np.loadtxt("C:\Sem VI\ELL409\Assignments\mnist_test.csv", delimiter=',')

    for k in range (epochs):
        abc=0
        batchsize = bs
        for ij in range(n):
            if(ij==n-1):
                batchsize = a_s
            sz = batchsize-abc

            delta = 0
            print(batchsize)
            for i in range(abc,batchsize):    
                train_pattern = np.transpose(train_data[i:i+1,1:])
                
                label = int(labels[i])
                hidden_l = sigmoid(np.dot(W1,train_pattern)+b1)

                output_l = softmax(np.dot(W2,hidden_l)+b2)
                d_output = np.zeros([10,1],dtype=float)
                d_output[label]=1
                delta = delta + d_output-output_l
                deltaT = delta.T
            hiddenT = hidden_l.T
            dl = np.dot(delta/sz,hiddenT)
            one = np.ones(hidden_l.shape)
            change = alpha*dl
            W2 = W2+change
            b2 = b2+alpha*(delta/sz)
            sigmo_d = np.multiply(hidden_l,(one-hidden_l))
            s2 = np.dot(W2.T,delta/sz)
            s3 = np.transpose(np.multiply(sigmo_d,s2))
            b1 = b1+alpha*(s3.T)
            dl2 = np.dot(train_pattern,s3)
            W1 = W1+alpha*(dl2.T)
            abc = abc+bs
            batchsize = batchsize+bs
    
    q=0
    for a in range(a_s,len(train_data)):
        pattern = np.transpose(train_data[a:a+1,1:])
        o_label = labels_test[a-a_s]
        hid = sigmoid(np.dot(W1,pattern)+b1)
        output = softmax(np.dot(W2,hid)+b2)
        oput = np.argmax(output)

        if o_label==oput:
            q+=1
    accuracy = 100*q/(len(train_data)-a_s)
    print(accuracy)
    xval.append(bs)
    yval.append(accuracy)
    bs = bs+1
plt.plot(xval,yval)


# In[ ]:




