def calculatedistance(X,Z=None):
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    
    if Z is None:
        Z=X;

    n,d1=X.shape
    m,d2=Z.shape

    s1 = np.sum(X**2, axis=1) ## X.dot(X.T)
    S = np.expand_dims(s1,1)
    
    
    r1 = np.sum(Z**2, axis=1) ## Z.dot(Z.T)
    r2 = np.expand_dims(r1,1)
    R = r2.T
    G = X.dot(Z.T)
    
    D2 = S - 2*G + R
    D3 = np.maximum(D2,0)
    D = np.sqrt(D3) 
    
    return D