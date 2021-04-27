#!/usr/bin/python
import numpy as np
import sys
import random

def name2features(name):
    """
    Take a name and create a feature. 
    The feature can be any length and can be binary, integer or real numbers. 
    But every name must generate the same length feature. 
    """
    
    d = 128 # number of hashing buckets
    v = np.zeros(d)
    name=name.lower() 
    
    # hash prefixes & suffixes - alexander -> [prefixa, prefixal, prefixale, suffixr, suffixer, suffixder]
    
    prefix_max = 3
    for m in range(prefix_max):
        prefix_string='prefix'+name[0:min(m+1,len(name))]
        random.seed(prefix_string)
        prefix_index = int(random.randint(0,d-1))
        v[prefix_index] = 1
    
    suffix_max = 3
    for m in range(suffix_max):
        suffix_string='suffix'+name[-1:-min(m+2,len(name)+1):-1]
        random.seed(suffix_string)
        suffix_index = int(random.randint(0,d-1))
        v[suffix_index] = 1
        
    return v
