def cart(xTr,yTr):
    """Builds a CART tree.
    
    The maximum tree depth is defined by "maxdepth" (maxdepth=2 means one split).
    Each example can be weighted with "weights".

    Args:
        xTr:      n x d matrix of data
        yTr:      n-dimensional vector

    Returns:
        tree: root of decision tree
    """
    n,d = xTr.shape
    maxdepth =5
    
    
    if len(yTr) == 0:
        return None
    elif all_same(yTr):
        #print("i am in same ytr")
        return TreeNode(None, None, None, None, yTr[0])
    elif len(xTr)<=2:
        #print("len is less 2")
        return TreeNode(None,None,None,None,np.mean(yTr))
    elif all_same(xTr):
        #print("i am in same all x")
        return TreeNode(None,None,None,None,majority_vote(yTr))
    else:
      
        fid, cut, loss = sqsplit(xTr,yTr)
      
        y_left = yTr[xTr[:, fid] <= cut]
        y_right = yTr[xTr[:, fid] > cut]
        
       
        x_left = xTr[xTr[:,fid]<=cut]
      
        x_right = xTr[xTr[:,fid]>cut]
       
        
        leftnode = cart(x_left,y_left)
        rightnode = cart(x_right,y_right)
        root= TreeNode(leftnode, rightnode,fid,cut,majority_vote(yTr))
  
        return root