def findknn(xTr,xTe,k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);
    
    Finds the k nearest neighbors of xTe in xTr.
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """

   
    D = calculatedistance(xTr,xTe)   #Distance of Xtr and Xte
    indices = np.argsort(D, axis=0)
    indices2 = indices[:k,:] #index of nearest k training data
   
    
    dists = np.sort(D,axis=0)
    dists2 = dists[:k,:]   ##distance to nearest k training data from test data
    return indices2,dists2