import numpy as np
import numbers

def create(vals):
    ADs = []
    num = len(vals)
    for i in range(num):
        val = vals[i]
        der = [0]*num
        der[i] = 1
        # print('val: {}, der: {}'.format(val,der))
        ADs.append(AutoDiff(val, der))
    return ADs

def stack(ADs):
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
        try: # assume exp is of AutoDiff type
        	return AutoDiff(self.val**exp.val,
        		(self.val**exp.val) * (self.der*exp.val/self.val + exp.der*np.log(self.val)))
        except AttributeError: # assume other is a number
        	return AutoDiff(self.val**exp, exp*(self.val**(exp-1))*self.der)
        	# if other is not a number, a TypeError will be raised

    def __rpow__(self,base):
        try: # assume exp is of AutoDiff type
        	return AutoDiff(base.val**self.val,
        		(base.val**self.val) * (base.der*self.val/base.val + self.der*np.log(base.val)))
        except AttributeError: # assume other is a number
       		return AutoDiff(base**self.val, np.log(base)*(base**self.val)*self.der)
       		# if other is not a number, a TypeError will be raised

    def __neg__(self):
        return AutoDiff(-self.val, -self.der)

    def __abs__(self):
        return AutoDiff(abs(self.val), (self.val/abs(self.val))*self.der)

    def __repr__(self):
        return("{0}({1},{2})".format(self.__class__.__name__, self.val,self.der))
    
    def __str__(self):
        return("AutoDiff Object, val: {0}, der: {1}".format(self.val,self.der))
    
    def __len__(self):
        return len(self.val)
    
    def __eq__(self, other):
        if self.val==other.val and self.der==other.der:
            return True
        else:
            return False
    
    def get_val(self):
        return self.val
    
    def get_der(self):
        return self.der

    
def sin(x):
    try:
        return AutoDiff(np.sin(x.val), np.cos(x.val)*x.der)
    except AttributeError:
        return np.sin(x)

def cos(x):
    try:
        return AutoDiff(np.cos(x.val), -np.sin(x.val)*x.der)
    except AttributeError:
        return np.cos(x)

def exp(x):
    try:
        return AutoDiff(np.exp(x.val), np.exp(x.val)*x.der)
    except AttributeError:
        return np.exp(x)

def log(x):
    try:
        if x.val <= 0:
            raise ValueError("Cannot take log of negative value")
        else:
            return AutoDiff(np.log(x.val), x.der/x.val)
    except AttributeError:
        if x <= 0:
            raise ValueError("Cannot take log of negative value")
        else:
            return np.log(x)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
