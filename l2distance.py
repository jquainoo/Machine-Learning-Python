import numpy as np
import innerproduct as innerproduct
def l2distance(X,Z=None):
    # function D=l2distance(X,Z)
    #
    # Computes the Euclidean distance matrix.
    # Syntax:
    # D=l2distance(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    #
    # call with only one input:
    # l2distance(X)=l2distance(X,X)
    #

    if Z is None:
        D=l2distance(X,X)
    else:  # case when there are two inputs (X,Z)
        G=innerproduct(X,Z)
        n=X.shape[0]
        m=Z.shape[0]
        d=X.shape[1]
        Xshape = X.shape
        Zshape = Z.shape
        X_ones = np.ones((1,d))
        Z_ones = np.ones((1,d))
        xi = np.dot(X_ones,np.transpose(np.square(X)))
        zj = np.dot(Z_ones, np.transpose(np.square(Z)))
        S=np.zeros((n,m))+xi
        R=np.zeros((n,m))+zj
        D2=abs(S-2*G+R)
        D=np.sqrt(D2)
    return D
    raise NotImplementedError()

