
def sqimpurity(yTr):
    """Computes the weighted variance of the labels
    
    Input:
        yTr:     n-dimensional vector of labels
    
    Output:
        impurity: weighted variance / squared loss impurity of this data set
    """
    
    ymean = np.mean(yTr)
    
    for y in yTr:
        impurity += (y-ymean)**2
    return impurity