import numpy as np
import numbers
import math

def create_f(vals):
    if np.array(vals).ndim == 0:
        return fAD(vals,[1])
    elif np.array(vals).ndim == 1:
        ADs = []
        num_var = len(vals)
        for i in range(num_var):
            val = vals[i]
            der = [0]*num_var
            der[i] = 1
            ADs.append(fAD(val, der))
        return ADs
    elif np.array(vals).ndim == 2:
        vals = np.array(vals)
        ADs = []
        num_var, num_dim = np.shape(vals)[0],np.shape(vals)[1]
        for i in range(num_var):
            AD_var = []
            for j in range(num_dim):
                val = vals[i,j]
                der = [0]*num_var
                der[i] = 1
                AD_var.append(fAD(val,der))
            ADs.append(stack_f(AD_var))
        return ADs
    elif np.array(vals).ndim > 2:
        raise ValueError('Input is at most 2D.')

def stack_f(ADs):
    new_val = []
    new_der = []
    for AD in ADs:
        for val in AD.val:
            new_val.append(val)
        for der in AD.der:
            new_der.append(der)
    new_AD = fAD(new_val,new_der)
    return new_AD

class fAD():
    def __init__(self,val,der=1):
        ## process val
        # check dimension
        if np.array(val).ndim > 1:
            raise ValueError('First argument cannot be 2D or higher.')
        val = np.array([val]).reshape(-1)
        if len(val) == 0:
            raise ValueError('First argument cannot be empty.')

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
            return fAD(self.val+other.val,self.der+other.der)
        except AttributeError: # assume other is a number
            return fAD(self.val+other,self.der)
            # if other is not a number, a TypeError will be raised

    def __radd__(self,other):
        try: # assume other is of AutoDiff type
            return fAD(self.val+other.val,self.der+other.der)
        except AttributeError: # assume other is a number
            return fAD(self.val+other,self.der)
            # if other is not a number, a TypeError will be raised

    def __sub__(self,other):
        try: # assume other is of AutoDiff type
            return fAD(self.val-other.val,self.der-other.der)
        except AttributeError: # assume other is a number
            return fAD(self.val-other,self.der)
            # if other is not a number, a TypeError will be raised

    def __rsub__(self,other):
        try: # assume other is of AutoDiff type
            return fAD(other.val-self.val,other.der-self.der)
        except AttributeError: # assume other is a number
            return fAD(other-self.val,-self.der)
            # if other is not a number, a TypeError will be raised


    def __mul__(self,other):
        try: # assume other is of AutoDiff type
             return fAD(self.val*other.val,mul_by_row(self.val,other.der)+mul_by_row(other.val,self.der))
        except AttributeError: # assume other is a number
            return fAD(self.val*other,self.der*other)
            # if other is not a number, a TypeError will be raised

    def __rmul__(self,other):
        try: # assume other is of AutoDiff type
            return fAD(self.val*other.val,mul_by_row(self.val,other.der)+mul_by_row(other.val,self.der))
        except AttributeError: # assume other is a number
            return fAD(self.val*other,self.der*other)
            # if other is not a number, a TypeError will be raised

    def __truediv__(self,other): # self/other
        try: # assume other is of AutoDiff type
             return fAD(self.val/other.val, mul_by_row(1/other.val,self.der)-mul_by_row(self.val/(other.val**2),other.der))
        except AttributeError: # assume other is a number
            return fAD(self.val/other,self.der/other)
            # if other is not a number, a TypeError will be raised

    def __rtruediv__(self,other): # other/self
        try: # assume other is of AutoDiff type
            return fAD(other.val/self.val, mul_by_row(1/self.val,other.der)-mul_by_row(other.val/(self.val**2),self.der))
        except AttributeError: # assume other is a number
            return fAD(other/self.val,mul_by_row(-other/(self.val**2),self.der))
            # if other is not a number, a TypeError will be raised

    def __pow__(self,exp):
        try: # assume exp is of AutoDiff type
        	return fAD(self.val**exp.val,
        		mul_by_row(self.val**exp.val,
                (mul_by_row(exp.val/self.val,self.der) + mul_by_row(np.log(self.val),exp.der))))
        except AttributeError: # assume other is a number
        	return fAD(self.val**exp, mul_by_row(exp*(self.val**(exp-1)),self.der))
        	# if other is not a number, a TypeError will be raised

    def __rpow__(self,base):
        try: # assume exp is of AutoDiff type
        	return fAD(base.val**self.val,
        		mul_by_row((base.val**self.val),
                (mul_by_row(self.val/base.val,base.der) + mul_by_row(np.log(base.val),self.der))))
        except AttributeError: # assume other is a number
       		return fAD(base**self.val, mul_by_row(np.log(base)*(base**self.val),self.der))
       		# if other is not a number, a TypeError will be raised

    def __neg__(self):
        return fAD(-self.val, -self.der)

    def __abs__(self):
        return fAD(abs(self.val), mul_by_row(self.val/abs(self.val),self.der))

    def __repr__(self):
        return("{0}({1},{2})".format(self.__class__.__name__, self.val,self.der))

    def __str__(self):
        return("fAD Object, val: {0}, der: {1}".format(self.val,self.der))

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
        if np.shape(self.val)[0] == 1:
            return self.val[0]
        else:
            return self.val

    def get_jac(self):
        if np.shape(self.der)[0] == 1 and np.shape(self.der)[1] == 1:
            return self.der[0,0]
        elif np.shape(self.der)[0] == 1 and np.shape(self.der)[1] > 1:
            return self.der[0]
        else:
            return self.der

def create_r(vals):
    if np.array(vals).ndim == 0:
        return rAD(vals)
    elif np.array(vals).ndim > 2:
        raise ValueError('Input is at most 2D.')
    else:
        ADs = [rAD(val) for val in vals]
        return ADs

def stack_r(vals,functions):
    jac = []
    for f in functions:
        vars = [rAD(val) for val in vals]
        f(*vars).outer()
        grad = [var.get_grad() for var in vars]
        jac.append(grad)
    return jac

class rAD:
    def __init__(self, vals):
        # check dimension of 'value'
        if np.array(vals).ndim > 1:
            raise ValueError('Input should be a scaler or a vector of numbers.')
        for i in np.array([vals]).reshape(-1):
            if not isinstance(i,numbers.Number):
                raise TypeError('Input should be a scaler or a vector of numbers.')
        self.val = np.array([vals]).reshape(-1,)
        self.children = []
        self.der = None


    def grad(self):
        if self.der is None:
            self.der = sum(w*a.grad() for w,a in self.children)
        return self.der


    def get_val(self):
        if np.shape(self.val)[0] == 1:
            return self.val[0]
        else:
            return self.val

    def get_grad(self):
        grad = self.grad()
        try:
            if np.shape(grad)[0] == 1: # gradient is a single value
                return grad[0]
            else: # gradient is an array
                return grad
        except IndexError: # 'outer' function
            return grad

    def __add__(self, other):
        try:
            ad = rAD(self.val + other.val)
            self.children.append((np.array([1.0]*len(self.val)), ad))
            other.children.append((np.array([1.0]*len(self.val)), ad))
            return ad
        except AttributeError:
            ad = rAD(self.val + other)
            self.children.append((np.array([1.0]*len(self.val)), ad))
            return ad

    def __radd__(self, other):
        try:
            ad = rAD(self.val + other.val)
            self.children.append((np.array([1.0]*len(self.val)), ad))
            other.children.append((np.array([1.0]*len(self.val)), ad))
            return ad
        except AttributeError:
            ad = rAD(self.val + other)
            self.children.append((np.array([1.0]*len(self.val)), ad))
            return ad

    def __sub__(self, other):
        try:
            ad = rAD(self.val - other.val)
            self.children.append((np.array([1.0]*len(self.val)), ad))
            other.children.append((np.array([-1.0]*len(self.val)), ad))
            return ad
        except AttributeError:
            ad = rAD(self.val - other)
            self.children.append((np.array([1.0]*len(self.val)), ad))
            return ad

    def __rsub__(self, other):
        try:
            ad = rAD(other.val - self.val)
            self.children.append((np.array([-1.0]*len(self.val)), ad))
            other.children.append((np.array([1.0]*len(self.val)), ad))
            return ad
        except AttributeError:
            ad = rAD(other - self.val)
            self.children.append((np.array([-1.0]*len(self.val)), ad))
            return ad

    def __mul__(self, other):
        try:
            ad = rAD(self.val * other.val)
            self.children.append((other.val, ad))
            other.children.append((self.val, ad))
            return ad
        except AttributeError:
            ad = rAD(self.val * other)
            self.children.append((np.array([other]*len(self.val)), ad))
            return ad

    def __rmul__(self, other):
        try:
            ad = rAD(self.val * other.val)
            self.children.append((other.val, ad))
            other.children.append((self.val, ad))
            return ad
        except AttributeError:
            ad = rAD(self.val * other)
            self.children.append((np.array([other]*len(self.val)), ad))
            return ad

    def __truediv__(self, other):
        try:
            ad = rAD(self.val / other.val)
            self.children.append((1/other.val, ad))
            other.children.append((-self.val/(other.val**2), ad))
            return ad
        except AttributeError:
            ad = rAD(self.val / other)
            self.children.append((np.array([1/other]*len(self.val)), ad))
            return ad

    def __rtruediv__(self, other):
        try:
            ad = rAD(other.val / self.val)
            self.children.append((-other.val/(self.val**2), ad))
            other.children.append((1/self.val, ad))
            return ad
        except AttributeError:
            ad = rAD(other / self.val)
            self.children.append((-other/(self.val**2), ad))
            return ad

    def __pow__(self, other):
        try:
            ad = rAD(self.val ** other.val)
            self.children.append((self.val**(other.val-1)*other.val, ad))
            other.children.append((self.val**other.val*np.log(self.val), ad))
            return ad
        except AttributeError:
            ad = rAD(self.val ** other)
            self.children.append((self.val**(other-1)*other, ad))
            return ad

    def __rpow__(self, other):
        try:
            ad = rAD(self.val ** other.val)
            self.children.append((other.val**self.val*np.log(other.val), ad))
            other.children.append((other.val**(self.val-1)*self.val, ad))
            return ad
        except AttributeError:
            ad = rAD(other ** self.val)
            self.children.append((other**self.val*np.log(other), ad))
            return ad

    def __neg__(self):
        new = rAD(-self.val)
        self.children.append((-1.0, new))
        return new

    def __abs__(self):
        new = rAD(abs(self.val))
        self.children.append((self.val/abs(self.val), new))
        return new

    def __str__(self):
        return "rAD Object, val: {0}, der: {1}".format(self.val, self.grad())

    def __eq__(self, other):
        if self.val == other.val and self.der == other.der:
            return True
        else:
            return False

    def __ne__(self, other):
        if self.val == other.val and self.der == other.der:
            return False
        else:
            return True

    def outer(self):
        self.der = 1.0

def sin(x):
    try: # x <- rAD
        ad = rAD(np.sin(x.val))
        x.children.append((np.cos(x.val),ad))
        return ad
    except AttributeError:
        try: # x <- fAD
            return fAD(np.sin(x.val), mul_by_row(np.cos(x.val),x.der))
        except AttributeError: # x <- numeric
            return np.sin(x)

def cos(x):
    try: # x <- rAD
        ad = rAD(np.cos(x.val))
        x.children.append((-np.sin(x.val),ad))
        return ad
    except AttributeError: 
        try: # x <- fAD
            return fAD(np.cos(x.val), mul_by_row(-np.sin(x.val),x.der))
        except AttributeError: # x <- numeric
            return np.cos(x)

def arcsin(x):
    try:
        #if x is an rAD object
        new = rAD(np.arcsin(x.val))
        x.children.append(((1/np.sqrt(1 - x.val*x.val)), new))
        return new
    except AttributeError:
        try:
            #if x is an fAD object
            return fAD(np.arcsin(x.val), mul_by_row(1/np.sqrt(1 - x.val*x.val),x.der))
        except AttributeError:
            #if x is a number
            return np.arcsin(x)

def arccos(x):
    try:
        #if x is an rAD object
        new = rAD(np.arccos(x.val))
        x.children.append(((-1/np.sqrt(1-x.val*x.val)), new))
        return new
    except AttributeError:
        try:
            #if x is an fAD object
            return fAD(np.arccos(x.val), mul_by_row((-1/np.sqrt(1-x.val*x.val)),x.der))
        except AttributeError:
            #if x is a number
            return np.arccos(x)
    
def arctan(x):
    try:
        #if x is an rAD object
        new = rAD(np.arctan(x.val))
        x.children.append(((1/(1+x.val*x.val)), new))
        return new
    except AttributeError:
        try:
            #if x is an fAD object
            return fAD(np.arctan(x.val), mul_by_row((1/(1+x.val*x.val)),x.der))
        except AttributeError:
            #if x is a number
            return np.arctan(x)

def sinh(x):
    try:
        #if x is an rAD object
        new = rAD(np.sinh(x.val))
        x.children.append((np.cosh(x.val), new))
        return new
    except AttributeError:
        try:
            #if x is an fAD object
            return fAD(np.sinh(x.val), mul_by_row(np.cosh(x.val),x.der))
        except AttributeError:
            #if x is a number
            return np.sinh(x)        

def exp(x):
    try:  # x <- rAD
        ad = rAD(np.exp(x.val))
        x.children.append((np.exp(x.val),ad))
        return ad
    except AttributeError: 
        try: # x <- fAD
            return fAD(np.exp(x.val), mul_by_row(np.exp(x.val),x.der))
        except AttributeError: # x <- numeric
            return np.exp(x)

def log(x,base=np.e):
    try: # x <- rAD
        if x.val <= 0:
            raise ValueError("Cannot take log of negative value")
        else:
            ad = rAD(math.log(x.val,base))
            x.children.append((1/(x.val*math.log(base)),ad))
            return ad
    except AttributeError:
        try: # x <- fAD
            if x.val <= 0:
                raise ValueError("Cannot take log of negative value")
            else:
                return fAD(math.log(x.val,base), mul_by_row(1/(x.val*math.log(base)),x.der))
        except AttributeError: # x <- numeric
            if x <= 0:
                raise ValueError("Cannot take log of negative value")
            else:
                return np.log(x)
    
def tan(x):
    try: #rAD
        ad = rAD(np.tan(x.val))
        x.children.append((1/(np.cos(x.val)**2),ad))
        return ad
    except AttributeError:
        try: #fAD
            return fAD(np.tan(x.val), mul_by_row(1/(np.cos(x.val)**2),x.der))
        except AttributeError:
            return np.tan(x) #numeric

def cosh(x):
    try:
        #if x is an rAD object
        new = rAD(np.cosh(x.val)) #
        x.children.append((np.sinh(x.val), new))
        return new
    except AttributeError:
        try:
            #if x is an fAD object
            return fAD(np.cosh(x.val), mul_by_row(np.sinh(x.val),x.der))
        except AttributeError:
            #if x is a number
            return np.cosh(x) 
def tanh(x):
    try:
        #if x is an rAD object
        new = rAD(np.tanh(x.val))
        x.children.append((1/(np.cosh(x.val)**2),new))
        return new
    except AttributeError:
        try:
            #if x is an fAD object
            return fAD(np.tanh(x.val), mul_by_row(1/(np.cosh(x.val)**2),x.der))
        except AttributeError:
            return np.tanh(x)
        
def sqrt(x):
    try: # reverse
        ad = rAD(x.val**0.5)
        x.children.append(((x.val**(-0.5))*0.5,ad))
        return ad
    except AttributeError:
        try: # forward
            return fAD(x.val**0.5, mul_by_row(0.5*(x.val**(-0.5)),x.der))
        except AttributeError:
            return x**0.5 #just a value 

def mul_by_row(val,der):
    if np.array(der).ndim <= 1:
        return val*der
    else:
        result = [val[i]*der[i] for i in range(len(val))]
        return np.array(result)

def reset_der(rADs):
    try:
        rADs.der = None
        rADs.children = []
    except AttributeError:
        for rAD in rADs:
            rAD.der = None
            rAD.children = []
