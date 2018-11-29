import numpy as np
import numbers

def create(vals):
    if np.array(vals).ndim == 0:
        return AutoDiff(vals,[1])
    if np.array(vals).ndim == 1:
        ADs = []
        num_var = len(vals)
        for i in range(num_var):
            val = vals[i]
            der = [0]*num_var
            der[i] = 1
            ADs.append(AutoDiff(val, der))
        return ADs
    if np.array(vals).ndim == 2:
        vals = np.array(vals)
        ADs = []
        num_var, num_dim = np.shape(vals)[0],np.shape(vals)[1]
        for i in range(num_var):
            AD_var = []
            for j in range(num_dim):
                val = vals[i,j]
                der = [0]*num_var
                der[i] = 1
                AD_var.append(AutoDiff(val,der))
            ADs.append(stack(AD_var))
        return ADs
    if np.array(vals).ndim > 2:
        raise ValueError('Input is at most 2D.')

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
    def __init__(self,val,der):
        ## process val
        # check dimension
        if np.array(val).ndim > 1:
            raise ValueError('First argument cannot be 2D or higher.')
        val = np.array([val]).reshape(-1) 
        if len(val) == 0:
            raise ValueError('First argument cannot be empty')

        # check variable type
        for i in val:
            if not isinstance(i,numbers.Number):
                raise TypeError('Arguments need to be consisted of numbers.')
        # store variable as attribute
        self.val = val

        ## process der
        # check dimension
        if len(self.val) == 1:
            ## scaler function
            if np.array(der).ndim <= 1 or np.shape(der)[0] == 1:
                der = np.array([[der]]).reshape(1,-1)
            else:
                raise ValueError('Input dimensions do not match.')
        elif len(self.val) > 1:
            ## vector function
            if np.shape(der)[0] == len(self.val):
                der = np.array([[der]]).reshape(len(self.val),-1)
            else:
                raise ValueError('Input dimensions do not match.')
        # check variable type
        for i in der.reshape(-1):
            if not isinstance(i,numbers.Number):
                raise TypeError('Arguments need to be consisted of numbers.')
        # store variable as attribute
        self.der = der
    
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

    def __ne__(self, other):
        if self.val!=other.val or self.der!=other.der:
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
