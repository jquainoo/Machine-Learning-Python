def knnclassifier(xTr,yTr,xTe,k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);
    
    k-nn classifier 
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    
    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    # fix array shapes
    yTr = yTr.flatten()

    
    indices, dists = findknn(xTr,xTe,k)
    s,d = xTe.shape
    
    
    vs = yTr[indices]
    preds = np.array([mode(vs[:,i])[0] for i in range (s)]).flatten()
    return preds