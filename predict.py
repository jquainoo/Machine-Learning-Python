def evaltree(root,xTe):
    """Evaluates xTe using decision tree root.
    
    Input:
        root: TreeNode decision tree
        xTe:  n x d matrix of data points
    
    Output:
        pred: n-dimensional vector of predictions
    """
  
    y_pred = []
    for inputs in xTe:
        prediction =  predict(root, inputs)
        y_pred.append(prediction)
        
    
    return y_pred
  
  
  
  
  
  
  def predict(root, X):
        
        """Predict class for a single sample."""
        #node = self.tree_
        while root.left:
            if X[root.feature] < root.cut:
                root = root.left
            else:
                root = root.right
        return root.prediction