import numpy as np
import numbers

def AD_create(vals):
    print(vals)
    ADs = []
    num = len(vals)
    for i in range(num):
        val = vals[i]
        der = [0]*num
        der[i] = 1
        print('val: {}, der: {}'.format(val,der))
        ADs.append(AutoDiff(val, der))
    return ADs

def AD_stack(ADs):
    new_val = []
    new_der = []
    for AD in ADs:
        for val in AD.val:
            new_val.append(val)
        for der in AD.der:
            new_der.append(der)
    new_AD = AutoDiff(new_val,new_der)
    return new_AD

class AutoDiff():
    def __init__(self,val,der=1):
        ## store inputs as arrays
        val = np.array([val]).reshape(-1) 
        der = np.array(der)
        ## check input type and dimension
        for i in val:
            if not isinstance(i,numbers.Number):
                raise TypeError('First argument needs to be consisted of numbers')
        for i in der.reshape(-1):
            if type(der[0])==list:
                raise ValueError('Input dimensions do not match!')
            if not isinstance(i,numbers.Number):
                raise TypeError('Second argument needs to be consisted of numbers')
        der = der.reshape(len(val),-1)
        if der.shape[0] != len(val):
            raise ValueError('Input dimensions do not match')

        self.val = val
        self.der = der
    
    def __add__(self,other):
        try: # assume other is of AutoDiffToy type
            return AutoDiff(self.val+other.val,self.der)
        except AttributeError: # assume other is a number
            return AutoDiff(self.val+other,self.der)
            # if other is not a number, a TypeError will be raised
    
    def __radd__(self,other):
        try: # assume other is of AutoDiffToy type
            return AutoDiff(self.val+other.val,self.der)
        except AttributeError: # assume other is a number
            return AutoDiff(self.val+other,self.der)
            # if other is not a number, a TypeError will be raised
    
    def __mul__(self,other):
        try: # assume other is of AutoDiffToy type
            return AutoDiff(self.val*other.val,self.val*other.der+self.der*other.val)
        except AttributeError: # assume other is a number
            return AutoDiff(self.val*other,self.der*other)
            # if other is not a number, a TypeError will be raised

    def __rmul__(self,other):
        try: # assume other is of AutoDiffToy type
            return AutoDiff(self.val*other.val,self.val*other.der+self.der*other.val)
        except AttributeError: # assume other is a number
            return AutoDiff(self.val*other,self.der*other)
            # if other is not a number, a TypeError will be raised