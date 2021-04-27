import numpy as np

def innerproduct(X,Z=None):
   # function innerproduct(X,Z)
   #
   # Computes the inner-product matrix.
   # Syntax:
   # D=innerproduct(X,Z)
   # Input:
   # X: nxd data matrix with n vectors (rows) of dimensionality d
   # Z: mxd data matrix with m vectors (rows) of dimensionality d
   #
   # Output:
   # Matrix G of size nxm
   # G[i,j] is the inner-product between vectors X[i,:] and Z[j,:]
   #
   # call with only one input:
   # innerproduct(X)=innerproduct(X,X)
   #
   if Z is None: # case when there is only one input (X)
       G=innerproduct(X,X)
   else: # case when there are two inputs (X,Z)
       G=np.dot(X,np.transpose(Z))
   return G
   raise NotImplementedError()

def innerprod_1():
   x = np.random.rand(700,10)
   print(x)
   test = (innerproduct(x).shape == 700,700)
   return test

def innerprod_2():
   print(abs(-2))

