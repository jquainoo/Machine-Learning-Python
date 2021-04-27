def sqsplit(xTr, yTr):
    """Finds the best feature, cut value, and loss value.
    
    Input:
        xTr:     n x d matrix of data points
        yTr:     n-dimensional vector of labels
    
    Output:
        feature:  index of the best cut's feature
        cut:      cut-value of the best cut
        bestloss: loss of the best cut
    """
    N,D = xTr.shape
    
    
    bestloss = np.inf
    feature = np.inf
    cut = np.inf

  
    idx   = np.argsort(xTr.T)
    for i, c in enumerate(xTr.T):
        list1 = np.array(c)[idx[i]]
        list2 = np.array(yTr)[idx[i]]
        if(all_same(list1)):
            continue
    
   
        for r in range(len(list1)-1):
            lftsquare=0.0
            rghtsquare =0.0
        
           
            if(list1[r] < list1[r+1]):
            
                if len(list2[:r+1])!=0:
                    lftsquare = sqimpurity(list2[:r+1])
                if len(list2[r+2:])!=0:
                    rghtsquare = sqimpurity(list2[r+1:])
               
                value = (list1[r]+list1[r+1])/2.0
              
                myloss = lftsquare + rghtsquare
                if myloss <=bestloss:
                    bestloss=myloss
                    cut = value
                    feature = i
    return feature, cut, bestloss