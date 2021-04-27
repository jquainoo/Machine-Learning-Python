def accuracy(truth,preds):
    """
    function output=accuracy(truth,preds)         
    Analyzes the accuracy of a prediction against the ground truth
    
    Input:
    truth = n-dimensional vector of true class labels
    preds = n-dimensional vector of predictions
    
    Output:
    accuracy = scalar (percent of predictions that are correct)
    """
    
    truth = truth.flatten()
    preds = preds.flatten()

  
    if len(truth)==0 and len(preds)==0:
        accuracy = 0
        return accuracy
    accuracy = np.mean(truth == preds)
    return accuracy
