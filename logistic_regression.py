#!/usr/bin/env python
# coding: utf-8

# In[74]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

# In[75]:

data_X=pd.read_csv('x.txt', sep='\ +', header=None,  engine='python')
data_Y=pd.read_csv('y.txt', sep='\ +', header=None, engine='python')

# In[76]:

data_X['Y']=data_Y[0].values
Xd=data_X[[0,1]].values
data_Y= data_Y.astype(int)

# In[77]:

Xd=np.hstack([np.ones((Xd.shape[0],1)), Xd])#
Yd=data_X['Y'].values

# In[78]:

#Initializing Newtons method with theta=0 
Parameters=[]
lim=1e9
iters=0
theta=np.zeros(Xd.shape[1])
while lim>1e-6:
    z=Yd*Xd.dot(theta)
    sigmoid = 1 / ( 1 + np . exp ( - z ) )
    delta_inv=np.mean((sigmoid-1)*Yd*Xd.T,axis=1)
    hessian=np.zeros((Xd.shape[1],Xd.shape[1]))
    for i in range(hessian.shape[0]):
        for j in range(hessian.shape[0]):
            if i<=j:
                hessian[i][j]=np.mean(sigmoid*(1-sigmoid)*Xd[:,i]*Xd[:,j])
                if i!=j:
                    hessian[j][i]=hessian[i][j]
                    
    delta=np.linalg.inv(hessian).dot(delta_inv)
    old_theta=theta.copy()
    theta-=delta
    Parameters.append(theta.copy())
    iters+=1
    lim = np.sum(np.abs(theta - old_theta))

print("Final Parameters - [{0},{1},{2}]".format(theta[0],theta[1],theta[2]))
print('It took {0} iterations'.format(iters))

#Plotting all the iterations 
ax = plt.axes()
data_X.query('Y==-1').plot.scatter(x=0, y=1, ax=ax, color='brown',label='negative')
data_X.query('Y==1').plot.scatter(x=0, y=1, ax=ax, color='green', label='positive')
x_vals = np.array([np.min(Xd[:,1]), np.max(Xd[:,1])])
for k, theta in enumerate(Parameters):
      targeted_para= (theta[0] + theta[1] * x_vals) / (- theta[2])
      plt.plot(x_vals,targeted_para)
plt.legend(loc=0)

# In[79]:

##part-c
#decision boundry fit by logistic regression
ax = plt.axes()
data_X.query('Y == -1').plot.scatter(x=0, y=1, ax=ax, color='red',marker='_', label='negative')
data_X.query('Y == 1').plot.scatter(x=0, y=1, ax=ax, color='blue', marker='+', label='positive')
plt.xlabel("x1")
plt.ylabel("x2")
x_vals = np.array([np.min(Xd[:,1]), np.max(Xd[:,1])])
targeted_para= (theta[0] + theta[1] * x_vals) / (- theta[2])
plt.plot(x_vals, targeted_para, 'g')
plt.legend(loc=0)
plt.figure(2, figsize=(15, 10))

# In[ ]:




