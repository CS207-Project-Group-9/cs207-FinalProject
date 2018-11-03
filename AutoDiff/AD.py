import numpy as np
import numbers

def AD_create(vals):
    ADs = []
    num = len(vals)
    for i in range(num):
        val = vals[i]
        der = [0]*num
        der[i] = 1
        # print('val: {}, der: {}'.format(val,der))
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
        ## store val in a 1D array
        val = np.array([val]).reshape(-1) 
        for i in val:
            if not isinstance(i,numbers.Number):
                raise TypeError('First argument needs to be consisted of numbers')
        self.val = val

        '''
        ## store der in a 2D array
        der = np.array([der])
        # check that der is not over 2D
        if der.ndim >= 3:
            raise ValueError('Second argument cannot be more than 2-D')
        # check that der dimension matches the val dimension
        try:
            der = der.reshape(len(val),-1)
        except ValueError:
            raise ValueError('Input dimensions do not match')
        # check if der is a 2D list of weird shape
        # e.g. [[1,2,3],[1,2]]
        if type(der[0]) == list:
            raise ValueError('Input dimensions do not match!')
        # check data type
        for i in der:
            for j in i:
                if not isinstance(j,numbers.Number):
                    raise TypeError('Second argument needs to be consisted of numbers')
        self.der = der
        '''

        ## store der in a 2D array
        ## der input is a single value
        if np.array(der).ndim == 0:
            # check data type
            if not isinstance(der,numbers.Number):
                raise TypeError('Second argument needs to be consisted of numbers')
            else:
                # store in 2D array
                self.der = np.array([[der]])
        
        ## der input is a vector
        elif np.array(der).ndim == 1:
            # check if der is a 2D list of weird shape
            # e.g. [[1,2,3],[1,2]]
            if type(der[0]) == list:
                raise ValueError('Input dimensions do not match!')
            # store in 2D array
            der = np.array([der]).reshape(len(self.val),-1)
            # check that der dimension matches the val dimension
            if len(self.val) != 1 and der.shape[1] != 1:
                raise ValueError('Input dimensions do not match')
            # check data type
            for i in der:
                for j in i:
                    if not isinstance(j,numbers.Number):
                        raise TypeError('Second argument needs to be consisted of numbers')
            self.der = der
            
        ## der input is a matrix
        elif np.array(der).ndim == 2:
            # store in 2D array:
            der = np.array(der)
            # check that der dimension matches the val dimension
            if der.shape[0] != len(self.val):
                raise ValueError('Input dimensions do not match')
            # check data type
            for i in der:
                for j in i:
                    if not isinstance(j,numbers.Number):
                        raise TypeError('Second argument needs to be consisted of numbers')
            self.der = der

        ## der input is high dimensional
        else:
            raise ValueError('Second argument cannot be more than 2-D')
        
    
    def __add__(self,other):
        try: # assume other is of AutoDiff type
            return AutoDiff(self.val+other.val,self.der+other.der)
        except AttributeError: # assume other is a number
            return AutoDiff(self.val+other,self.der)
            # if other is not a number, a TypeError will be raised
    
    def __radd__(self,other):
        try: # assume other is of AutoDiff type
            return AutoDiff(self.val+other.val,self.der+other.der)
        except AttributeError: # assume other is a number
            return AutoDiff(self.val+other,self.der)
            # if other is not a number, a TypeError will be raised
    
    def __sub__(self,other):
        try: # assume other is of AutoDiff type
            return AutoDiff(self.val-other.val,self.der-other.der)
        except AttributeError: # assume other is a number
            return AutoDiff(self.val-other,self.der)
            # if other is not a number, a TypeError will be raised
    
    def __rsub__(self,other):
        try: # assume other is of AutoDiff type
            return AutoDiff(other.val-self.val,other.der-self.der)
        except AttributeError: # assume other is a number
            return AutoDiff(other-self.val,-self.der)
            # if other is not a number, a TypeError will be raised


    def __mul__(self,other):
        try: # assume other is of AutoDiff type
             return AutoDiff(self.val*other.val,self.val*other.der+self.der*other.val)
        except AttributeError: # assume other is a number
            return AutoDiff(self.val*other,self.der*other)
            # if other is not a number, a TypeError will be raised

    def __rmul__(self,other):
        try: # assume other is of AutoDiff type
            return AutoDiff(self.val*other.val,self.val*other.der+self.der*other.val)
        except AttributeError: # assume other is a number
            return AutoDiff(self.val*other,self.der*other)
            # if other is not a number, a TypeError will be raised

    def __truediv__(self,other): # self/other
        try: # assume other is of AutoDiff type
             return AutoDiff(self.val/other.val, self.der/other.val-self.val*other.der/(other.val**2))
        except AttributeError: # assume other is a number
            return AutoDiff(self.val/other,self.der/other)
            # if other is not a number, a TypeError will be raised

    def __rtruediv__(self,other): # other/self
        try: # assume other is of AutoDiff type
            return AutoDiff(other.val/self.val,other.val/self.der-other.val*self.der/(self.val**2))
        except AttributeError: # assume other is a number
            return AutoDiff(other/self.val,-other*self.der/(self.val**2))
            # if other is not a number, a TypeError will be raised

    def __pow__(self,exp):
        if not isinstance(exp,numbers.Number):
            raise TypeError('Exponent needs to be a number.')
        return AutoDiff(self.val**exp, exp*(self.val**(exp-1))*self.der)

    def __rpow__(self,base):
        if not isinstance(base,numbers.Number):
            raise TypeError('Base needs to be a number.')
        return AutoDiff(base**self.val, np.log(base)*(base**self.val)*self.der)

    def __neg__(self):
        return AutoDiff(-self.val, -self.der)

    def sin(self):
        return AutoDiff(np.sin(self.val), np.cos(self.val)*self.der)

    def cos(self):
        return AutoDiff(np.cos(self.val), -np.sin(self.val)*self.der)

    ## The functions below are not required for Milestone 2

    def __abs__(self):
        return


    def log(self):
        return AutoDiff(np.log(self.val), self.der/self.val)



