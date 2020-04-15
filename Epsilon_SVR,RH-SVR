#This file contains code for both Epsilon-SVR(including Sklearn method) and RH-SVR
# In[140]:


import numpy as np 
from numpy import linalg
import pandas as pd
import cvxopt as co
from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt
import math

#Importing data set
train_data = pd.read_csv(r'SVR.csv')
train_data = np.matrix(train_data)
#MEDV values
MEDV = train_data[:,-1]
MEDV = np.array(MEDV).reshape(506,)
#Normalised data set (X-Values)
train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
train_data = np.matrix(train_data)
#Features
features = train_data[:,:13]

#Kernels
def polynomial_kernel(x,y,gamma,deg): 
    return (gamma+np.dot(x,y))**deg 
def rbf_kernel(x,y,gamma): 
    return np.exp(-gamma*(linalg.norm(x-y)**2)) 


#Epsilon Support Vector Regression using cvxopt
    #kernel=0->Linear
    #kernel=1->Polynomial
    #kernel=2->RBF
def epsilon_svr(features,MEDV,ltrain,epsilon,C,kernel,gamma,deg):
    
    #P matrix
    c = np.matrix(np.zeros(2*ltrain)).T
    for j in range(ltrain):
        e = [0,0]
        for i in range (ltrain):
            a = np.matrix([[1.0,-1.0],[-1.0,1.0]])
            if(kernel==0):
                a = a*np.asscalar(np.dot(np.matrix(features[i]),np.matrix(features[j]).T))
            if(kernel==1):
                a = a*np.asscalar(polynomial_kernel(np.matrix(features[i]),np.matrix(features[j]).T,gamma,deg))
            if(kernel==2):
                a = a*np.asscalar(rbf_kernel(np.matrix(features[i]),np.matrix(features[j]),gamma))
            e = np.vstack((e,a))
        e = e[1:,:]
        c = np.hstack((c,e))
    
    c = c[:,1:]
    P = co.matrix(c,tc='d')



    #q matrix
    q = np.matrix([0])
    for i in range(ltrain):
        a = np.matrix([[epsilon-MEDV[i]],[epsilon+MEDV[i]]])
        q = np.vstack((q,a))
    q = q[1:,:]
    q = co.matrix(q,tc='d')
    
    #A matrix
    A = np.matrix([0])
    for i in range(ltrain):
        A = np.hstack((A,np.matrix([1])))
        A = np.hstack((A,np.matrix([-1])))
    A = A[:,1:]
    A = co.matrix(A,tc='d')
   
    #b matrix
    b = np.matrix([0])
    b= co.matrix(b,tc='d')

    #G matrix
    a = np.zeros((2*ltrain,2*ltrain))
    np.fill_diagonal(a, -1)
    G = a
    a = np.zeros((2*ltrain,2*ltrain))
    np.fill_diagonal(a, 1)
    G = np.vstack((G,a))
    G = co.matrix(G,tc='d')
    
    #h matrix
    a = np.zeros((2*ltrain,1))
    h = a
    a = C*np.ones((2*ltrain,1))
    h = np.vstack((h,a))
    h = co.matrix(h,tc='d')
    
    #Solution using solvers(cvxopt)
    sol = solvers.qp(P,q,G,h,A,b)
    alpha = np.array(sol['x'])
    
    #Bias Term
    bias = np.asscalar(np.array(sol['y']))
    
    return alpha,bias
    

#K fold cross validation
    #kernel=0->Linear
    #kernel=1->Polynomial
    #kernel=2->RBF
def cross_validation(epsilon,C,kernel,gamma,deg):
    #5 fold
    total = (len(features))
    size = int(len(features)/5)
    training_length = total-size
    training__length = 4*size
    RMSE_final = 0
    RMSE = 0
    for pq in range(5):
        if(pq==0):
            traininglength = training__length
            features_train = features[0:traininglength,:]
            features_test = features[traininglength:,:]
            MEDV_train = MEDV[0:traininglength]
            MEDV_test = MEDV[traininglength:]
        else:
            traininglength = training_length
            features_train = np.vstack((features[pq*size:,:],features[:(pq-1)*size,:]))
            features_test = features[(pq-1)*size:pq*size]
            MEDV_train = np.hstack((MEDV[pq*size:],MEDV[:(pq-1)*size]))
            MEDV_test = MEDV[(pq-1)*size:pq*size]
        
        testlength = total-traininglength
        
        result = epsilon_svr(features_train,MEDV_train,traininglength,epsilon,C,kernel,gamma,deg)
        alpha = result[0]
        bias = result[1]
        
        RMSE1 = 0
        predicted_values = []
        for i in range (testlength):
            wt=0
            for j in range (traininglength):
                if(kernel==0):
                    wt = np.asscalar(wt + (alpha[2*j]-alpha[2*j+1])*np.asscalar(np.dot(np.matrix(features_train[j]),np.matrix(features_test[i]).T)))
                if(kernel==1):
                    wt = np.asscalar(wt + (alpha[2*j]-alpha[2*j+1])*np.asscalar(polynomial_kernel(np.matrix(features_train[j]),np.matrix(features_test[i]).T,gamma,deg)))
                if(kernel==2):
                    wt =  np.asscalar(wt + (alpha[2*j]-alpha[2*j+1])*np.asscalar(rbf_kernel(np.matrix(features_train[j]),np.matrix(features_test[i]),gamma)))
            wt = wt+bias
            predicted_values.append(wt)
            RMSE1 = RMSE1 + (wt-MEDV_test[i])**2
        RMSE = RMSE + (RMSE1/testlength)**(0.5)
    
    return RMSE/5


# In[122]:


#Variation in RMSE with change in epsilon
    
#Linear Kernel
C=1
xval = []
yval = []
a=10
i=0
while (i<5):
    xval.append(math.log10(a))
    yval.append(cross_validation(a,C,0,0,0))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(Epsilon)')
plt.xlabel('log(Epsilon)')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

xval = []
yval = []
a=0.1
while (a<1.1):
    xval.append(a)
    yval.append(cross_validation(a,C,0,0,0))
    a=a+0.1
plt.plot(xval,yval)
plt.title('RMSE vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

#Polynomial Kernel
#Gamma=1
#Degree=2
C=1
xval = []
yval = []
a=10
i=0
while (i<5):
    xval.append(math.log10(a))
    yval.append(cross_validation(a,C,1,1,2))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(Epsilon)')
plt.xlabel('log(Epsilon)')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

xval = []
yval = []
a=5
while (a<16):
    xval.append(a)
    yval.append(cross_validation(a,C,1,1,2))
    a=a+1
plt.plot(xval,yval)
plt.title('RMSE vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()


#Polynomial Kernel
#Gamma=1
#Degree=3
C=1
xval = []
yval = []
a=10
i=0
while (i<5):
    xval.append(math.log10(a))
    yval.append(cross_validation(a,C,1,1,3))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(Epsilon)')
plt.xlabel('log(Epsilon)')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

xval = []
yval = []
a=0
while (a<10):
    xval.append(a)
    yval.append(cross_validation(a,C,1,1,3))
    a=a+1
plt.plot(xval,yval)
plt.title('RMSE vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()


#RBF Kernel
#Gamma=2
C=1
xval = []
yval = []
a=10
i=0
while (i<5):
    xval.append(math.log10(a))
    yval.append(cross_validation(a,C,2,2,0))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(Epsilon)')
plt.xlabel('log(Epsilon)')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

xval = []
yval = []
a=0
while (a<10):
    xval.append(a)
    yval.append(cross_validation(a,C,2,2,0))
    a=a+1
plt.plot(xval,yval)
plt.title('RMSE vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()


#Variation in RMSE with change in C

#linear Kernel
epsilon=0.8
xval = []
yval = []
a=10000000
i=0
while (i<11):
    xval.append(math.log10(a))
    yval.append(cross_validation(epsilon,a,0,0,0))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(C)')
plt.xlabel('log(C)')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

epsilon=0.8
xval = []
yval = []
a=50000
while (a<550000):
    xval.append(a)
    yval.append(cross_validation(epsilon,a,0,0,0))
    a=a+50000
plt.plot(xval,yval)
plt.title('RMSE vs C')
plt.xlabel('C')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()


#Polynomial Kernel
#Gamma=1
#Degree=2
Epsilon = 6
xval = []
yval = []
a=1000000
i=0
while (i<8):
    xval.append(math.log10(a))
    yval.append(cross_validation(Epsilon,a,1,1,2))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(C)')
plt.xlabel('log(C)')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

Epsilon = 6
xval = []
yval = []
a=5000
while (a<50000):
    xval.append(a)
    yval.append(cross_validation(Epsilon,a,1,1,2))
    a=a+5000
plt.plot(xval,yval)
plt.title('RMSE vs C')
plt.xlabel('C')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

#Polynomial Kernel
#Gamma=1
#Degree=3
Epsilon = 5
xval = []
yval = []
a=1000000
i=0
while (i<8):
    xval.append(math.log10(a))
    yval.append(cross_validation(Epsilon,a,1,1,3))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(C)')
plt.xlabel('log(C)')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()


Epsilon = 5
xval = []
yval = []
a=500
while (a<5500):
    xval.append(a)
    yval.append(cross_validation(Epsilon,a,1,1,3))
    a=a+500
plt.plot(xval,yval)
plt.title('RMSE vs C')
plt.xlabel('C')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

#RBF Kernel
#Gamma=2
epsilon=1
xval = []
yval = []
a=1000000
i=0
while (i<8):
    xval.append(math.log10(a))
    yval.append(cross_validation(epsilon,a,2,2,0))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(C)')
plt.xlabel('log(C)')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

epsilon=1
xval = []
yval = []
a=5000
while (a<50000):
    xval.append(a)
    yval.append(cross_validation(epsilon,a,2,2,0))
    a=a+5000
plt.plot(xval,yval)
plt.title('RMSE vs C')
plt.xlabel('C')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()


#Variation in RMSE with change in gamma (Polynomial and RBF)

#Those values of Epsilon and C are chosen which gave minimum RMSE in previous plots 


#Polynomial Kernel
#Degree=2
Epsilon = 6
C = 20000
xval = []
yval = []
a=100
i=0
while (i<5):
    xval.append(math.log10(a))
    yval.append(cross_validation(Epsilon,C,1,a,2))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(gamma)')
plt.xlabel('log(gamma)')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()



Epsilon = 6
C = 20000
xval = []
yval = []
a=0
while (a<5.5):
    xval.append(a)
    yval.append(cross_validation(Epsilon,C,1,a,2))
    a=a+0.5
plt.plot(xval,yval)
plt.title('RMSE vs gamma')
plt.xlabel('gamma')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

#Polynomial Kernel
#Degree=3
Epsilon = 5
C = 2500
xval = []
yval = []
a=10
i=0
while (i<5):
    xval.append(math.log10(a))
    yval.append(cross_validation(Epsilon,C,1,a,3))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(gamma)')
plt.xlabel('log(gamma)')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

Epsilon = 5
C = 2500
xval = []
yval = []
a=0
while (a<5.5):
    xval.append(a)
    yval.append(cross_validation(Epsilon,C,1,a,3))
    a=a+0.5
plt.plot(xval,yval)
plt.title('RMSE vs gamma')
plt.xlabel('gamma')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

#RBF Kernel
Epsilon = 1
C=20000
xval = []
yval = []
a=10000
i=0
while (i<8):
    xval.append(math.log10(a))
    yval.append(cross_validation(Epsilon,C,2,a,0))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(gamma)')
plt.xlabel('log(gamma)')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()

Epsilon = 1
C=20000
xval = []
yval = []
a=0.5
while (a<6.5):
    xval.append(a)
    yval.append(cross_validation(Epsilon,C,2,a,0))
    a=a+0.5
plt.plot(xval,yval)
plt.title('RMSE vs gamma')
plt.xlabel('gamma')
plt.ylabel('Root Mean Sqaured Error(RMSE)')
plt.show()


# In[181]:


#Epsilon Support Vector Regression using Sklearn

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

#Finding minimum RSE using Sklearn
# 0->Linear
# 1->Polynomial
# 2->RBF
def score_RMSE(kernel1):
    #Finding Best Parameters and Hyperparameters using Grid Search  
    if(kernel1==0):
        gsc = GridSearchCV(estimator=SVR(kernel='linear'),param_grid={'epsilon':np.arange(0.0,10.0,1.0),'C': [10**i for i in np.arange(0.0,6.0,1.0)]},cv=5, scoring='neg_mean_squared_error', verbose=1,n_jobs=-1)
    if(kernel1==1):
        gsc = GridSearchCV(estimator=SVR(kernel='poly'),param_grid={'epsilon':np.arange(0.0,10.0,1.0),'C': [10**i for i in np.arange(0.0,6.0,1.0)],'gamma': np.arange(0.0,3.0,0.5),'degree':np.arange(2.0,4.0,1.0)},cv=5, scoring='neg_mean_squared_error', verbose=1,n_jobs=-1)
    if(kernel1==2):    
        gsc = GridSearchCV(estimator=SVR(kernel='rbf'),param_grid={'epsilon':np.arange(0.0,10.0,1.0),'C': [10**i for i in np.arange(0.0,6.0,1.0)],'gamma': np.arange(0.0,3.0,0.5)},cv=5, scoring='neg_mean_squared_error', verbose=1,n_jobs=-1)
    grid_result = gsc.fit(features, MEDV)
    best_parameters = grid_result.best_params_
    print("Best Hyperparameters and Kernel Parameters ",best_parameters)
    C1 = best_parameters['C']
    epsilon1 = best_parameters['epsilon']
    if(kernel1>0):
        gamma1 = best_parameters['gamma']
    if(kernel1==1):
        degree1 = best_parameters['degree']
    #Calculating the Root Mean Squared Error using K-fold Cross-Validation
    #K=5
    RMSE = []
    kf = KFold(n_splits = 5, shuffle = False)
    for train_index,test_index in kf.split(features):
        X_train,X_test,y_train,y_test = features[train_index],features[test_index],MEDV[train_index],MEDV[test_index]
        if(kernel1==0):
            clf = SVR(C=C1,epsilon=epsilon1,kernel='linear')
        if(kernel1==1):
            clf = SVR(C=C1,epsilon=epsilon1,kernel='poly',degree=degree1,gamma=gamma1)
        if(kernel1==2):
            clf = SVR(C=C1,epsilon=epsilon1,kernel='rbf',gamma=gamma1)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        RMSE.append((mean_squared_error(y_test,y_pred))**(0.5))
    RMSE_mean = np.mean(RMSE)
    return RMSE_mean

#Linear Kernel
print(score_RMSE(0))

#Polynomial Kernel
print(score_RMSE(1))

#RBF kernel
print(score_RMSE(2))


# In[291]:


#Reduced Convex Hull Support Vector Regression using cvxopt
    #kernel=0->Linear
    #kernel=1->Polynomial
    #kernel=2->RBF
def rhsvm_train(features,MEDV,ltrain,epsilon,D,kernel,gamma,deg):
    
    #P matrix
    c1 = np.matrix(np.zeros(ltrain))
    for i in range(ltrain):
        c12 = np.zeros((2,ltrain))
        c12[0][i]=1
        c12[1][i]=-1
        c1 = np.vstack((c1,c12))
    c1 = c1[1:,:]
    if(kernel==0):
        P = np.dot(features,features.T)
    if(kernel==1):
        P = np.zeros((ltrain,ltrain))
        for i in range(ltrain):
            for j in range(ltrain):
                P[i][j]= polynomial_kernel(features[i],features[j].T,gamma,deg)
    if(kernel==2):
        P = np.zeros((ltrain,ltrain))
        for i in range(ltrain):
            for j in range(ltrain):
                P[i][j]= rbf_kernel(features[i],features[j],gamma)
    P = P + np.dot(np.matrix(MEDV).T,np.matrix(MEDV))
    P = np.dot(c1,P)
    P = np.dot(P,c1.T)
    P = co.matrix(P,tc='d')
    
    #q matrix
    q =  2*epsilon*np.dot(c1,np.matrix(MEDV).T)
    q = co.matrix(q,tc='d')
    
    #A matrix
    A = np.zeros((2,2*ltrain))
    for i in range(ltrain):
        A[0][2*i]=1
        A[1][2*i+1]=1
    
    A = co.matrix(A,tc='d')
    
    #b matrix
    b = np.ones((2,1))
    b = co.matrix(b,tc='d')
    
    #G matrix
    a1 = np.zeros((2*ltrain,2*ltrain))
    np.fill_diagonal(a1, -1)
    a2 = np.zeros((2*ltrain,2*ltrain))
    np.fill_diagonal(a2, 1)
    G = np.vstack((a1,a2))
    G = co.matrix(G,tc='d')
    
    #h matrix
    a1 = np.zeros((2*ltrain,1))
    a2 = D*np.ones((2*ltrain,1))
    h = np.vstack((a1,a2))
    h = co.matrix(h,tc='d')
    
    sol = solvers.qp(P,q,G,h,A,b)
    
    uv = np.matrix(sol['x'])
    
    #delta
    delta = np.dot(uv.T,c1)
    delta = np.asscalar(np.dot(delta,np.matrix(MEDV).T)) + 2*epsilon
    
    #u_bar
    a1 = np.zeros((ltrain,2*ltrain))
    for i in range(ltrain):
        a1[i][2*i] = 1
    u_bar = np.dot(a1,uv)
    u_bar = u_bar/delta
    
    #v_bar
    a1 = np.zeros((ltrain,2*ltrain))
    for i in range(ltrain):
        a1[i][2*i+1] = 1
    v_bar = np.dot(a1,uv)
    v_bar = v_bar/delta
    
    #bias
    bias = np.dot(uv.T,c1)
    if(kernel==0):
        bias = np.dot(bias,np.dot(features,features.T))
    if(kernel==1):
        bias12 = np.zeros((ltrain,ltrain))
        for i in range(ltrain):
            for j in range(ltrain):
                bias12[i][j]= polynomial_kernel(features[i],features[j].T,gamma,deg)
        bias = np.dot(bias,bias12)
    if(kernel==2):
        bias12 = np.zeros((ltrain,ltrain))
        for i in range(ltrain):
            for j in range(ltrain):
                bias12[i][j]= rbf_kernel(features[i],features[j],gamma)
        bias = np.dot(bias,bias12)
    c2 = np.matrix(np.zeros(ltrain))
    for i in range(ltrain):
        c12 = np.zeros((2,ltrain))
        c12[0][i]=1
        c12[1][i]=1
        c2 = np.vstack((c2,c12))
    c2 = c2[1:,:]
    c2 = c2.T
    bias = np.dot(bias,np.dot(c2,uv))
    bias = bias/(2*delta)
    bias = np.asscalar(bias)
    add_term = np.dot(c2,uv)
    add_term = np.dot(add_term.T,np.matrix(MEDV).T)
    add_term = 0.5*np.asscalar(add_term)
    bias = bias + add_term
    
    return v_bar,u_bar,bias
    
    

#K fold cross validation
    #kernel=0->Linear
    #kernel=1->Polynomial
    #kernel=2->RBF
def cross_validation_rh(epsilon,D,kernel,gamma,deg):
    #5 fold
    total = (len(features))
    size = int(len(features)/5)
    training_length = total-size
    training__length = 4*size
    RMSE_final = 0
    RMSE = 0
    for pq in range(5):
        if(pq==0):
            traininglength = training__length
            features_train = features[0:traininglength,:]
            features_test = features[traininglength:,:]
            MEDV_train = MEDV[0:traininglength]
            MEDV_test = MEDV[traininglength:]
        else:
            traininglength = training_length
            features_train = np.vstack((features[pq*size:,:],features[:(pq-1)*size,:]))
            features_test = features[(pq-1)*size:pq*size]
            MEDV_train = np.hstack((MEDV[pq*size:],MEDV[:(pq-1)*size]))
            MEDV_test = MEDV[(pq-1)*size:pq*size]
        
        testlength = total-traininglength
        
        result = rhsvm_train(features_train,MEDV_train,traininglength,epsilon,D,kernel,gamma,deg)
        v_bar = result[0]
        u_bar = result[1]
        bias = result[2]
        RMSE1 = 0
        predicted_values = []
    
        for i in range(testlength):
            sum = 0
            for j in range(traininglength):
                if(kernel==0):
                    kernel_val = np.asscalar(np.dot(features_train[j],features_test[i].T))
                if(kernel==1):
                    kernel_val = np.asscalar(polynomial_kernel(features_train[j],features_test[i].T,gamma,deg))
                if(kernel==2):
                    kernel_val = np.asscalar(rbf_kernel(features_train[j],features_test[i],gamma))
                sum = sum + np.asscalar((v_bar[j]-u_bar[j]))*kernel_val
            sum = sum + bias
            predicted_values.append(sum)
            RMSE1 = RMSE1 + (sum-MEDV_test[i])**2
        RMSE = RMSE + (RMSE1/testlength)**(0.5)
    return RMSE/5
    
    


# In[208]:


#Variation in RMSE with change in epsilon
    
#Linear Kernel

D=0.5
xval = []
yval = []
a=10
i=0
while (i<5):
    xval.append(math.log10(a))
    yval.append(cross_validation_rh(a,D,0,0,0))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(Epsilon)')
plt.xlabel('log(Epsilon)')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

D=0.5
xval = []
yval = []
a=0.5
while (a<5.5):
    xval.append(a)
    yval.append(cross_validation_rh(a,D,0,0,0))
    a=a+0.5
plt.plot(xval,yval)
plt.title('RMSE vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

xval=[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
plt.plot(xval,yval)
plt.title('RMSE vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

#Polynomial Kernel
#Gamma=1
#Degree=2

D=0.5
xval = []
yval = []
a=10
i=0
while (i<5):
    xval.append(math.log10(a))
    yval.append(cross_validation_rh(a,D,1,1,2))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(Epsilon)')
plt.xlabel('log(Epsilon)')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()  


D=0.5
xval = []
yval = []
a=0.5
while (a<5.5):
    xval.append(a)
    yval.append(cross_validation_rh(a,D,1,1,2))
    a=a+0.5
plt.plot(xval,yval)
plt.title('RMSE vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

#Polynomial Kernel
#Gamma=1
#Degree=3
D=0.5
xval = []
yval = []
a=100
i=0
while (i<6):
    xval.append(math.log10(a))
    yval.append(cross_validation_rh(a,D,1,1,3))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(Epsilon)')
plt.xlabel('log(Epsilon)')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()  


D=0.5
xval = []
yval = []
a=5
while (a<16):
    xval.append(a)
    yval.append(cross_validation_rh(a,D,1,1,3))
    a=a+1
plt.plot(xval,yval)
plt.title('RMSE vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

#RBF Kernel
#Gamma=2
D=0.5
xval = []
yval = []
a=100
i=0
while (i<5):
    xval.append(math.log10(a))
    yval.append(cross_validation_rh(a,D,2,2,0))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(Epsilon)')
plt.xlabel('log(Epsilon)')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

D=0.5
xval = []
yval = []
a=1
while (a<11):
    xval.append(a)
    yval.append(cross_validation_rh(a,D,2,2,0))
    a=a+1
plt.plot(xval,yval)
plt.title('RMSE vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

#Variation in RMSE with change in D
    
#Linear Kernel
Epsilon=3
xval = []
yval = []
a=1
i=0
while (i<3):
    xval.append(math.log10(a))
    yval.append(cross_validation_rh(Epsilon,a,0,0,0))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(D)')
plt.xlabel('log(D)')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

Epsilon=3
xval = []
yval = []
a=0.005
while (a<0.05):
    xval.append(a)
    yval.append(cross_validation_rh(Epsilon,a,0,0,0))
    a=a+0.005
plt.plot(xval,yval)
plt.title('RMSE vs D')
plt.xlabel('D')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()


#Polynomial Kernel
#Degree=2
#Gamma=1
Epsilon=1
xval = []
yval = []
a=1
i=0
while (i<3):
    xval.append(math.log10(a))
    yval.append(cross_validation_rh(Epsilon,a,1,1,2))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(D)')
plt.xlabel('log(D)')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

Epsilon=1
xval = []
yval = []
a=0.05
while (a<0.55):
    xval.append(a)
    yval.append(cross_validation_rh(Epsilon,a,1,1,2))
    a=a+0.05
plt.plot(xval,yval)
plt.title('RMSE vs D')
plt.xlabel('D')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()



#Polynomial Kernel
#Degree=3
#Gamma=1
Epsilon=8
xval = []
yval = []
a=1
i=0
while (i<3):
    xval.append(math.log10(a))
    yval.append(cross_validation_rh(Epsilon,a,1,1,3))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(D)')
plt.xlabel('log(D)')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

Epsilon=8
xval = []
yval = []
a=0.05
while (a<0.55):
    xval.append(a)
    yval.append(cross_validation_rh(Epsilon,a,1,1,3))
    a=a+0.05
plt.plot(xval,yval)
plt.title('RMSE vs D')
plt.xlabel('D')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

#Rbf Kernel
#Gamma=2
Epsilon=8
xval = []
yval = []
a=1
i=0
while (i<3):
    xval.append(math.log10(a))
    yval.append(cross_validation_rh(Epsilon,a,2,2,0))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(D)')
plt.xlabel('log(D)')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

Epsilon=8
xval = []
yval = []
a=0.5
while (a<1):
    xval.append(a)
    yval.append(cross_validation_rh(Epsilon,a,2,2,0))
    a=a+0.05
plt.plot(xval,yval)
plt.title('RMSE vs D')
plt.xlabel('D')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()



#Variation in RMSE with change in gamma (Polynomial and RBF)

#Those values of Epsilon and D are chosen which gave minimum RMSE in previous plots 

#Polynomial Kernel
#Degree=2
Epsilon = 1
D = 0.25
xval = []
yval = []
a=100
i=0
while (i<5):
    xval.append(math.log10(a))
    yval.append(cross_validation_rh(Epsilon,D,1,a,2))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(gamma)')
plt.xlabel('log(gamma)')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()


Epsilon = 1
D = 0.25
xval = []
yval = []
a=0.4
while (a<2.2):
    xval.append(a)
    yval.append(cross_validation_rh(Epsilon,D,1,a,2))
    a=a+0.2
plt.plot(xval,yval)
plt.title('RMSE vs gamma')
plt.xlabel('gamma')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()



#Polynomial Kernel
#Degree=3
Epsilon = 8
D = 0.1
xval = []
yval = []
a=1000
i=0
while (i<5):
    xval.append(math.log10(a))
    yval.append(cross_validation_rh(Epsilon,D,1,a,3))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(gamma)')
plt.xlabel('log(gamma)')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

Epsilon = 8
D = 0.1
xval = []
yval = []
a=50
while (a<160):
    xval.append(a)
    yval.append(cross_validation_rh(Epsilon,D,1,a,3))
    a=a+10
plt.plot(xval,yval)
plt.title('RMSE vs gamma')
plt.xlabel('gamma')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()



#RBF Kernel
Epsilon = 8
D = 0.9
xval = []
yval = []
a=1000
i=0
while (i<6):
    xval.append(math.log10(a))
    yval.append(cross_validation_rh(Epsilon,D,2,a,0))
    a=a/10
    i=i+1
plt.plot(xval,yval)
plt.title('RMSE vs log(gamma)')
plt.xlabel('log(gamma)')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

Epsilon = 8
D = 0.9
xval = []
yval = []
a=0.5
while (a<5.5):
    xval.append(a)
    yval.append(cross_validation_rh(Epsilon,D,2,a,0))
    a=a+0.5
plt.plot(xval,yval)
plt.title('RMSE vs gamma')
plt.xlabel('gamma')
plt.ylabel('Root Mean Squared Error(RMSE)')
plt.show()

