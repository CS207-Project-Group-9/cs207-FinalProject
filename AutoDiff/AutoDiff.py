import numpy as np
import numbers
import math

def create_f(vals):
    '''
    create_f(values)
    
    Create a forward-mode autodiff object.

    Parameters
    --------------
    values: array_like
        input variable values for automatic differentiation.
        Allows for up to 2-dimensional input.

    Returns
    --------------
    out: forward-mode automatic differentiation object satisfying the specific requirements.

    
    Examples
    --------------
    >>>> a = AutoDiff.create_f(3) #single variable, 0 dimension: 
    >>>> a.val
        array([3])
    >>>> a.der
        array([[1]])
    >>>> a, b, c = AutoDiff.create_f([1, 2, 3]) #multiple variables, 1 dimension
    >>>> a.val
        array([1])
    >>>> b.val
        array([2])
    >>>> c.val
        array([3])
    >>>> a.der
        array([[1, 0, 0]])
    >>>> b.der
        array([[0, 1, 0]])
    >>>> c.der
        array([[0, 0, 1]])
    >>>> a, b = AutoDiff.create_f([[1, 2],[3, 4]]) # multiple variables, 2 dimensions
    >>>> a.val
        array([1, 2])
    >>>> b.val
        array([3, 4])
    >>>> a.der
        array([[1, 0], [1, 0]])
    >>>> b.der
        array([[0, 1], [0, 1]])   
    '''
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
    '''
    stack_f(objects)
    
    Stack forward-mode autodiff objects.
    
    Parameters
    --------------
    objects: array_like
        input forward-mode autodiff objects as initiated by create_f()
        *dimensions of all objects must be the same*
                
    Returns
    --------------
    out: a forward-mode autodiff object.
        Values of forward-mode autodiff objects are stacked and returned as a vector.
        Derivatives of the objects are returned in a matrix.

    Examples
    --------------
    Functions to differentiate:
    f1 = 2a + 3b + c
    f2 = 5m * 2n + z

    >>>> a, b, c = create_f([1, 2, 3])
    >>>> m, n, z = create_f([4, 5, 6])
    >>>> f1 = 2*a + 3*b + c
    >>>> f2 = 5*m * 2*n + z

    stack f1, f2
    >>>> functions = stack_f([f1, f2])
    >>>> functions.val
        array([ 11, 206])
    >>>> functions.der
        array([[ 2,  3,  1], [50, 40,  1]])   
    '''
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
    '''
    fAD(value, derivative = 1)

    Create a forward-mode autodiff object.

    Parameters
    --------------
    value: number, or array_like if multiple values
        input variable values for differentiation.
        *Allows only 1-dimensional input of values, for 2-dimensional input, use create_f*

    derivative: optional for single value input
        must be defined when there are multiple values for differentiation.

    Attributes
    --------------
    val: array, shape of (1, n_values)
        n_values determined by length of value input

    der: array
        shape determined by input shape of derivatives

    Returns
    --------------
    out: a forward-mode autodiff object

    Examples
    --------------
    single value input:
    >>>> a = fAD(5.0)
    >>>> a.val
        array([5.0])
    >>>> a.der
        array([[1]])

    multiple values input:
    >>>> a = fAD([1,5], [[1,0],[0,1]])
    >>>> a.val
        array([1, 5])
    >>>> a.der
        array([[1, 0], [0, 1]])
    '''   
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
        '''
        Support addition between:
        1. forward autodiff objects
        2. a forward autodiff object and a number

        Returns
        --------------
        out: sum of two autodiff objects, or sum of an autodiff object and a number, as an autodiff object
        
        Example
        --------------
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> z = 3.0
        >>>> s1 = x + y
        >>>> s1.val
            array([12.0])
        >>>> s1.der
            array([[1, 1]])
        >>>> s2 = x + z
        >>>> s2.val
            array([8.0])
        >>>> s2.der
            array([[1, 0]])
        '''
        try: # assume other is of AutoDiff type
            return fAD(self.val+other.val,self.der+other.der)
        except AttributeError: # assume other is a number
            return fAD(self.val+other,self.der)
            # if other is not a number, a TypeError will be raised

    def __radd__(self,other):
        '''
        Support addition between:
        1. forward autodiff objects
        2. a number and a forward autodiff object

        Returns
        --------------
        out: sum of two autodiff objects, or sum of a number and an autodiff object, as an autodiff object

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> z = 3.0
        >>>> s1 = y + x
        >>>> s1.val
            array([12.0])
        >>>> s1.der
            array([[1, 1]])
        >>>> s2 = z + y
        >>>> s2.val
            array([10.0])
        >>>> s2.der
            array([[0, 1]])
        '''
        try: # assume other is of AutoDiff type
            return fAD(self.val+other.val,self.der+other.der)
        except AttributeError: # assume other is a number
            return fAD(self.val+other,self.der)
            # if other is not a number, a TypeError will be raised

    def __sub__(self,other):
        '''
        Support subtraction between:
        1. forward autodiff objects
        2. a forward autodiff object and a number

        Returns
        --------------
        out: difference between two autodiff objects, or difference between
            an autodiff object and a number, as an autodiff object

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> z = 3.0
        >>>> s1 = y - x
        >>>> s1.val
            array([2.0])
        >>>> s1.der
            array([[-1, 1]])
        >>>> s2 = x - z
        >>>> s2.val
            array([2.0])
        >>>> s2.der
            array([[1, 0]])
        '''
        try: # assume other is of AutoDiff type
            return fAD(self.val-other.val,self.der-other.der)
        except AttributeError: # assume other is a number
            return fAD(self.val-other,self.der)
            # if other is not a number, a TypeError will be raised

    def __rsub__(self,other):
        '''
        Support subtraction between:
        1. forward autodiff objects
        2. a number and a forward autodiff object

        Returns
        --------------
        out: difference between two autodiff objects,
            or difference between a number and an autodiff object, as an autodiff object

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> m = 10.0
        >>>> s1 = y - x
        >>>> s1.val
            array([2.0])
        >>>> s1.der
            array([[-1, 1]])
        >>>> s2 = m - x
        >>>> s2.val
            array([5.0])
        >>>> s2.der
            array([[-1, 0]])
        '''
        try: # assume other is of AutoDiff type
            return fAD(other.val-self.val,other.der-self.der)
        except AttributeError: # assume other is a number
            return fAD(other-self.val,-self.der)
            # if other is not a number, a TypeError will be raised


    def __mul__(self,other):
        '''
        Support multiplication of:
        1. forward autodiff objects
        2. a forward autodiff object and a number

        Returns
        --------------
        out: product two autodiff objects, or product of
            an autodiff object and a number, as an autodiff object

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> z = 3.0
        >>>> m1 = x*y
        >>>> m1.val
            array([35.0])
        >>>> m1.der
            array([[7.0, 5.0]])
        >>>> m2 = x*z
        >>>> m2.val
            array([15.0])
        >>>> m2.der
            array([[3.0, 0]])
        '''
        try: # assume other is of AutoDiff type
             return fAD(self.val*other.val,mul_by_row(self.val,other.der)+mul_by_row(other.val,self.der))
        except AttributeError: # assume other is a number
            return fAD(self.val*other,self.der*other)
            # if other is not a number, a TypeError will be raised

    def __rmul__(self,other):
        '''
        Support multiplication of:
        1. forward autodiff objects
        2. a number and a forward autodiff 

        Returns
        --------------
        out: product two autodiff objects, or product of a number and
            an autodiff object, as an autodiff object

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> z = 3.0
        >>>> m1 = y*x
        >>>> m1.val
            array([35.0])
        >>>> m1.der
            array([[7.0, 5.0]])
        >>>> m2 = z*y
        >>>> m2.val
            array([21.0])
        >>>> m2.der
            array([[0, 3.0]])
        '''
        try: # assume other is of AutoDiff type
            return fAD(self.val*other.val,mul_by_row(self.val,other.der)+mul_by_row(other.val,self.der))
        except AttributeError: # assume other is a number
            return fAD(self.val*other,self.der*other)
            # if other is not a number, a TypeError will be raised

    def __truediv__(self,other): # self/other
        '''
        Support division between:
        1. forward autodiff objects
        2. a forward autodiff and a number

        Returns
        --------------
        out: quotient of two autodiff objects divided by one another,
            or quotient of an autodiff object divided by a number, as an autodiff object

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([4.0, 8.0])
        >>>> z = 2.0
        >>>> d1 = y/x
        >>>> d1.val
            array([2.0])
        >>>> d1.der
            array([[-0.5, 0.25]])
        >>>> d2 = x/z
        >>>> d2.val
            array([2.0])
        >>>> d2.der
            array([[0.5, 0]])
        '''
        try: # assume other is of AutoDiff type
             return fAD(self.val/other.val, mul_by_row(1/other.val,self.der)-mul_by_row(self.val/(other.val**2),other.der))
        except AttributeError: # assume other is a number
            return fAD(self.val/other,self.der/other)
            # if other is not a number, a TypeError will be raised

    def __rtruediv__(self,other): # other/self
        '''
        Support division between:
        1. forward autodiff objects
        2. a number and a forward autodiff object

        Returns
        --------------
        out: quotient of two autodiff objects divided by one another,
            or quotient of a number divided by an autodiff object, as an autodiff object

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([4.0, 8.0])
        >>>> z = 2.0
        >>>> d1 = y/x
        >>>> d1.val
            array([2.0])
        >>>> d1.der
            array([[-0.5, 0.25]])
        >>>> d2 = z/y
        >>>> d2.val
            array([0.25])
        >>>> d2.der
            array([[0, -0.03125]])
        '''
        try: # assume other is of AutoDiff type
            return fAD(other.val/self.val, mul_by_row(1/self.val,other.der)-mul_by_row(other.val/(self.val**2),self.der))
        except AttributeError: # assume other is a number
            return fAD(other/self.val,mul_by_row(-other/(self.val**2),self.der))
            # if other is not a number, a TypeError will be raised

    def __pow__(self,exp):
        '''
        Support exponentiation of a forward autodiff object

        Returns
        --------------
        out: the power of an autodiff object, with the autodiff object as base,
            and either a number, or another autodiff object as the exponent, as an autodiff object

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([2.0, 3.0])
        >>>> z = 5.0
        >>>> p1 = (x*y)**z
        >>>> p1.val
            array([7776.0])
        >>>> p1.der
            array([[19440., 12960.]])
        >>>> p2 = x**y
        >>>> p2.val
            array([8.0])
        >>>> p2.der
            array([[12., 5.54517744]])
        '''
        try: # assume exp is of AutoDiff type
        	return fAD(self.val**exp.val,
        		mul_by_row(self.val**exp.val,
                (mul_by_row(exp.val/self.val,self.der) + mul_by_row(np.log(self.val),exp.der))))
        except AttributeError: # assume other is a number
        	return fAD(self.val**exp, mul_by_row(exp*(self.val**(exp-1)),self.der))
        	# if other is not a number, a TypeError will be raised

    def __rpow__(self,base):
        '''
        Support exponentiation of a forward autodiff object

        Returns
        --------------
        out: the power of an autodiff object, with either a number of an autodiff object as base,
            and an autodiff object as the exponent, as a forward autodiff object

        Example
        --------------
        >>>> x, y = AutoDiff.create_f(np.array([1.0, 2.0]))
        >>>> z = 5.0
        >>>> p1 = z ** (a*b)
        >>>> p1.val
            array([25.0])
        >>>> p1.der
            array([[80.47189562, 40.23594781]])
        '''
        try: # assume exp is of AutoDiff type
        	return fAD(base.val**self.val,
        		mul_by_row((base.val**self.val),
                (mul_by_row(self.val/base.val,base.der) + mul_by_row(np.log(base.val),self.der))))
        except AttributeError: # assume other is a number
       		return fAD(base**self.val, mul_by_row(np.log(base)*(base**self.val),self.der))
       		# if other is not a number, a TypeError will be raised

    def __neg__(self):
        '''
        Returns
        --------------
        out: the negative, or the opposite, of the autodiff object as a forward autodiff object

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([2.0, 8.0])
        >>>> n1 = -(x/y)
        >>>> n1.val
            array([-0.25])
        >>>> n1.der
            array([[-0.125, 0.03125]])
        '''
        return fAD(-self.val, -self.der)

    def __abs__(self):
        '''
        Returns
        --------------
        out: the absolute of the autodiff object as a forward autodiff object

        Example
        --------------
        >>>> a = AutoDiff.fAD(-8,1)
        >>>> b = abs(a)
        >>>> b.val
            array([8])
        >>>> b.der
            array([[-1.]])
        '''
        return fAD(abs(self.val), mul_by_row(self.val/abs(self.val),self.der))

    def __repr__(self):
        '''
        Returns
        --------------
        out: 'fAD(values, derivatives)'
            outputs autodiff object values, and partial derivatives

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> f = 4*x + y
        >>>> f
            fAD(27.0,[4 1])
        '''
        return "{0}({1},{2})".format(self.__class__.__name__, self.get_val(), self.get_jac())

    def __str__(self):
        '''
        Returns
        --------------
        out: "Forward-mode AutoDiff Object, value(s): values, partial derivative(s): derivatives" 
            outputs autodiff object values, and partial derivatives.

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> f = 4*x + y
        >>>> f
            'Forward-mode AutoDiff Object, value(s): 27.0, partial derivative(s): [4, 1]'
        '''
        return "Forward-mode AutoDiff Object, value(s): {0}, partial derivative(s): {1}".format(self.get_val(), self.get_jac())

    def __len__(self):
        '''
        Returns
        --------------
        out: number of variable values 

        Example
        --------------
        >>>> a, b = AutoDiff.create_f([2.0, 8.0])
        >>>> c = AutoDiff.stack_f([a,b])
        >>>> assert len(x) == 2
        '''
        return len(self.val)

    def __eq__(self, other):
        '''
        Allow comparisons between two forward autodiff objects
    
        Returns
        --------------
        out: True if two forward autodiff objects match in terms of values and partial derivatives

        Example
        --------------
        >>>> a = AutoDiff.fAD(8.0,1)
        >>>> b = AutoDiff.fAD(8.0,1)
        >>>> c = AutoDiff.fAD(5.0,1)
        >>>> assert a == b
        >>>> assert (a == c) == False
        '''
        if self.val==other.val and self.der==other.der:
            return True
        else:
            return False

    def __ne__(self, other):
        '''
        Allow comparisons between two forward autodiff objects
    
        Returns
        --------------
        out: True if two forward autodiff objects do not match in terms of
            values and/or partial derivatives

        Example
        --------------
        >>>> a = AutoDiff.fAD(8.0,1)
        >>>> b = AutoDiff.fAD(8.0,1)
        >>>> c = AutoDiff.fAD(5.0,1)
        >>>> assert a != b
        >>>> assert (a != c) == False
        '''
        if self.val!=other.val or self.der!=other.der:
            return True
        else:
            return False

    def get_val(self):
        '''
        fAD.get_val()

        Get values of differentiated object.
    
        Returns
        --------------
        out: numeric, or array_like
            function values as a result of supported operations (e.g. multiplication)

        Example
        --------------
        single function:
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> f = 4*x + y
        >>>> f.get_val()
            27.0

        multiple functions:
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> f1 = 4*x + y
        >>>> f2 = x**3 - y
        >>>> f = AutoDiff.stack_f([f1, f2])
        >>>> f.get_val()
            array([ 27., 118.])
        '''
        if np.shape(self.val)[0] == 1:
            return self.val[0]
        else:
            return self.val

    def get_jac(self):
        '''
        fAD.get_val()

        Get the Jacobian matrix of partial derivatives.
    
        Returns
        --------------
        out: array_like (vector for univariate operations, matrix for multivariate operations)
            partial derivatives with respect to function(s) as a result of supported operations (e.g. multiplication)

        Example
        --------------
        single function:
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> f = 4*x + y
        >>>> f.get_jac()
            array([4, 1])

        multiple functions:
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> f1 = 4*x + y
        >>>> f2 = x**3 - y
        >>>> f = AutoDiff.stack_f([f1, f2])
        >>>> f.get_jac()
            array([[ 4.,  1.], [75., -1.]])
        '''
        if np.shape(self.der)[0] == 1 and np.shape(self.der)[1] == 1:
            return self.der[0,0]
        elif np.shape(self.der)[0] == 1 and np.shape(self.der)[1] > 1:
            return self.der[0]
        else:
            return self.der

def create_r(vals):
    '''
    create_r(values)
    
    Create a reverse-mode autodiff object.

    Parameters
    --------------
    values: numeric, or array_like
        input variable values for automatic differentiation.
        input can be a single number for univariate operations.
        For multivariate operations, input values as an array.
        Allows for up to 2-dimensional input.
        *This method allows for simultaneous variable assignments.* 

    Returns
    --------------
    out: reverse-mode automatic differentiation object
        satisfying the specific requirements.

    
    Examples
    --------------
    import function:
    >>>> from AutoDiff import create_r
    
    single variable, 0 dimension:
    >>>> a = AutoDiff.create_r(2.0)
    >>>> f = AutoDiff.sin(a)
    >>>> f.outer()
    >>>> f.get_val() #outputs function value
        0.9092974268256817
    >>>> a.get_grad() #outputs df/da
        -0.4161468365471424

    multiple variables, 1 dimension: 
    >>>> a, b, c = create_r([1, 2, 3])
    >>>> f = 2*a + b**3 +AutoDiff.cos(c)
    >>>> f.outer()
    >>>> f.get_val() #outputs function value
        9.010007503399555
    >>>> a.get_grad() #outputs df/da
        2.0
    >>>> b.get_grad() #outputs df/db
        12.0
    >>>> c.get_grad() #outputs df/dc
        -0.14112001

    multiple variables, 2 dimensions:
    >>>> a, b = create_r([[1, 2],[3, 4]])
    >>>> f = 2*a + b**3
    >>>> f.outer()
    >>>> f.get_val() #outputs function value
        array([29, 68])
    >>>> a.get_grad() #df/da
        array([2., 2.])
    >>>> b.get_grad() #df/db
        array([27., 48.])   
    '''
    if np.array(vals).ndim == 0:
        return rAD(vals)
    elif np.array(vals).ndim > 2:
        raise ValueError('Input is at most 2D.')
    else:
        ADs = [rAD(val) for val in vals]
        return ADs

class rAD:
    '''
    rAD(value)

    Create a reverse-mode autodiff object.

    Parameters
    --------------
    value: number, or array_like if multiple values
        input variable values for differentiation.
        *Allows only 1-dimensional input of values, for 2-dimensional input, use create_r*

    Attributes
    --------------
    val: array, shape of (1, n_values)
        n_values determined by length of value input

    der: default to None for input variables.
        Use outer() to resert outer function derivative.

    Returns
    --------------
    out: a reverse-mode autodiff object

    Examples
    --------------
    single value input:
    >>>> a = rAD(5.0)
    >>>> f = 2**a
    >>>> f.outer()
    >>>> f.get_val() #output function value
        32.0
    >>>> a.get_grad() #output df/da
        22.18070977791825

    multiple values input:
    >>>> a = rAD([5.0, 3.2])
    >>>> f = 2**a
    >>>> f.outer()
    >>>> f.get_val() #output function value
        array([32.        ,  9.18958684])
    >>>> a.get_grad() #output df/da
        array([22.18070978,  6.36973621])
    '''   
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
        '''
        rAD.grad()

        Get the gradient of the variable.
    
        Returns
        --------------
        out: array_like (vector for single-value operations, matrix for multi-value operations)
            gradient of variable with respect to function.
            *calling variable.grad() before variable.der will update
            derivatives of variable from None to its gradient with respect to function.
 
        Example
        --------------
        single variable:
        >>>> a= AutoDiff.rAD([5.0])
        >>>> f = 4*a
        >>>> f.outer()
        >>>> a.grad()
            array([4.])

        multiple variables:
        >>>> a= AutoDiff.rAD([5.0, 7.0])
        >>>> f = 4*a 
        >>>> f.outer()
        >>>> a.grad()
            array([4., 4.])
        '''
        if self.der is None:
            self.der = sum(w*a.grad() for w,a in self.children)
        return self.der


    def get_val(self):
        '''
        rAD.get_val()

        Get values of differentiated object.
    
        Returns
        --------------
        out: numeric, or array_like
            function values as a result of supported operations (e.g. multiplication)

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> f = 4*x + y
        >>>> f.outer()
        >>>> f.get_val()
            27.0
        '''
        if np.shape(self.val)[0] == 1:
            return self.val[0]
        else:
            return self.val

    def get_grad(self):
        '''
        fAD.get_grad()

        Get the gradient of variable.
    
        Returns
        --------------
        out: array_like (vector for single-value operations, matrix for multi-value operations)
            gradient of variable with respect to function.
            *calling variable.grad() before variable.der will update
            derivatives of variable from None to its gradient with respect to function.*
            *must call get_grad() for individual variables, and not for the function*
            
        Example
        --------------
        >>>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>>> f = 4*a + 3**b
        >>>> f.outer()
        >>>> a.get_grad()
            array([4., 4.])
        >>>> b.get_grad()
            array([29.66253179, 88.98759538])
        '''
        grad = self.grad()
        if np.shape(grad)[0] == 1:
            return grad[0]
        else:
            return grad

    def __add__(self, other):
        '''
        Support addition between:
        1. reverse autodiff objects
        2. a reverse autodiff object and a number

        Returns
        --------------
        out: sum of two autodiff objects, or sum of an autodiff object and a number, as an autodiff object
        
        Example
        --------------
        >>>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>>> f = 4*(a+b) + 3**(b+5)
        >>>> f.outer()
        >>>> a.get_grad()
            array([4., 4.])
        >>>> b.get_grad()
            array([ 7211.99522595, 21627.98567785])
        >>>> f.get_val()
            array([ 6577, 19707])
        '''
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
        '''
        Support addition between:
        1. reverse autodiff objects
        2. a number and a reverse autodiff object

        Returns
        --------------
        out: sum of two autodiff objects, or sum of a number and a reverse autodiff object, as an autodiff object
        
        Example
        --------------
        >>>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>>> f = 4*(b+a) + 3**(5+b)
        >>>> f.outer()
        >>>> a.get_grad()
            array([4., 4.])
        >>>> b.get_grad()
            array([ 7211.99522595, 21627.98567785])
        >>>> f.get_val()
            array([ 6577, 19707])
        '''
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
        '''
        Support subtraction between:
        1. reverse autodiff objects
        2. a reverse autodiff object and a number

        Returns
        --------------
        out: difference between two autodiff objects, or difference between
            an autodiff object and a number, as an autodiff object

        Example
        --------------
        >>>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>>> f = AutoDiff.sin(a)-6*(b-2)
        >>>> f.outer()
        >>>> a.get_grad()
            array([ 0.54030231, -0.41614684])
        >>>> b.get_grad()
            array([-6., -6.])
        >>>> f.get_val()
            array([ -5.15852902, -11.09070257])
        '''
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
        '''
        Support subtraction between:
        1. reverse autodiff objects
        2. a number and a reverse autodiff object

        Returns
        --------------
        out: difference between two autodiff objects, or difference between
            a number and a reverse autodiff object, as an autodiff object

        Example
        --------------
        >>>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>>> f = AutoDiff.cos(a)-6*(8-b)
        >>>> f.outer()
        >>>> a.get_grad()
            array([-0.84147098, -0.90929743])
        >>>> b.get_grad()
            array([-6., -6.])
        >>>> f.get_val()
            array([-29.45969769, -24.41614684])
        '''
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
        '''
        Support multiplication of:
        1. reverse autodiff objects
        2. a reverse autodiff object and a number

        Returns
        --------------
        out: product two autodiff objects, or product of
            an autodiff object and a number, as an autodiff object

        Example
        --------------
        >>>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>>> f = AutoDiff.cos(a)-2*(a*b)
        >>>> f.outer()
        >>>> a.get_grad()
            array([-6.84147098, -8.90929743])
        >>>> b.get_grad()
            array([-2., -4.])
        >>>> f.get_val()
            array([ -5.45969769, -16.41614684])
        '''
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
        '''
        Support multiplication of:
        1. reverse autodiff objects
        2. a number and a reverse autodiff object

        Returns
        --------------
        out: product two autodiff objects, or product of
            a number and a reverse autodiff object, as an autodiff object

        Example
        --------------
        >>>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>>> f = AutoDiff.cos(a)-2*(b*5)
        >>>> f.outer()
        >>>> a.get_grad()
            array([-0.84147098, -0.90929743])
        >>>> b.get_grad()
            array([-10., -10.])
        >>>> f.get_val()
            array([-29.45969769, -40.41614684])
        '''
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
        '''
        Support division between:
        1. reverse autodiff objects
        2. a reverse autodiff and a number

        Returns
        --------------
        out: quotient of two autodiff objects divided by one another,
            or quotient of an autodiff object divided by a number, as an autodiff object

        Example
        --------------
        >>>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>>> f = AutoDiff.tan(a/b)-2*(b/5)
        >>>> f.outer()
        >>>> a.get_grad()
            array([0.37329717, 0.3246116 ])
        >>>> b.get_grad()
            array([-0.52443239, -0.5623058 ])
        >>>> f.get_val()
            array([-0.85374645, -1.05369751])
        '''
        try:
            ad = rAD(self.val / other.val)
            self.children.append((1/other.val, ad))
            other.children.append((-self.val/(other.val**2), ad))
            return ad
        except AttributeError:
            ad = rAD(self.val / other)
            self.children.append((1/other, ad))
            return ad

    def __rtruediv__(self, other):
        '''
        Support division between:
        1. reverse autodiff objects
        2. a number and a reverse division between

        Returns
        --------------
        out: quotient of two autodiff objects divided by one another,
            or quotient of a number and a reverse autodiff object, as an autodiff object

        Example
        --------------
        >>>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>>> f = AutoDiff.tan(a/b)-(8/b)
        >>>> f.outer()
        >>>> a.get_grad()
            array([0.37329717, 0.3246116 ])
        >>>> b.get_grad()
            array([0.7644565, 0.3376942])
        >>>> f.get_val()
            array([-2.32041312, -1.45369751])
        '''
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
        '''
        Support exponentiation of a reverse autodiff object

        Returns
        --------------
        out: the power of an autodiff object, with the autodiff object as base,
            and either a number, or another autodiff object as the exponent, as an autodiff object

        Example
        --------------
        >>>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>>> f = AutoDiff.tan(a/b)-b**3
        >>>> f.outer()
        >>>> a.get_grad()
            array([0.37329717, 0.3246116 ])
        >>>> b.get_grad()
            array([-27.12443239, -48.1623058 ])
        >>>> f.get_val()
            array([-26.65374645, -63.45369751])
        '''
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
        '''
        Support exponentiation of a reverse autodiff object

        Returns
        --------------
        out: the power of an autodiff object, with the autodiff object as base,
            and either a number, or another autodiff object as the exponent, as an autodiff object

        Example
        --------------
        >>>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>>> f = AutoDiff.tan(a/b)-2*(b/5)
        >>>> f.outer()
        >>>> a.get_grad()
            array([0.37329717, 0.3246116 ])
        >>>> b.get_grad()
            array([-0.52443239, -0.5623058 ])
        >>>> f.get_val()
            array([-0.85374645, -1.05369751])
        '''
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
        '''
        Returns
        --------------
        out: the negative, or the opposite, of the autodiff object as a reverse autodiff object

        Example
        --------------
        >>>> x, y = AutoDiff.create_r([2.0, 8.0])
        >>>> f = -x/y
        >>>> f.outer()
        >>>> x.get_grad()
            -0.125
        >>>> y.get_grad()
            0.03125
        >>>> f.get_val()
            -0.25
        '''
        new = rAD(-self.val)
        self.children.append((-1.0, new))
        return new

    def __abs__(self):
        '''
        Returns
        --------------
        out: the absolute of the autodiff object as a reverse autodiff object

        Example
        --------------
        >>>> a = AutoDiff.rAD(-8)
        >>>> f = abs(a) + 6
        >>>> f.outer()
        >>>> f.get_val()
            14
        >>>> a.get_grad()
            -1.0          
        '''
        new = rAD(abs(self.val))
        self.children.append((self.val/abs(self.val), new))
        return new

    def __str__(self):
        '''
        Returns
        --------------
        out: "Reverse AutoDiff Object, value(s): {0}, gradient: {1}"
            outputs autodiff object values, and gradient.
            *when print(outer function), the gradient is 1.0, please print(variables)
            to output gradient of variable with respect to function"

        Example
        --------------
        >>>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>>> f = 4*x + y
        >>>> f
            'Reverse AutoDiff Object, value(s): [27.], gradient: 1.0'
        '''
        return "Reverse AutoDiff Object, value(s): {0}, gradient: {1}".format(self.val, self.grad())

    def __eq__(self, other):
        '''
        Allow comparisons between two reverse autodiff objects
    
        Returns
        --------------
        out: True if two reverse autodiff objects match in terms of values and gradient

        Example
        --------------
        >>>> a = AutoDiff.rAD(8.0)
        >>>> b = AutoDiff.rAD(8.0)
        >>>> c = AutoDiff.rAD(5.0)
        >>>> assert a == b
        >>>> assert (a == c) == False
        '''
        if self.val == other.val and self.der == other.der:
            return True
        else:
            return False

    def __ne__(self, other):
        '''
        Allow comparisons between two reverse autodiff objects
    
        Returns
        --------------
        out: True if two reverse autodiff objects do not match in terms of values and gradient

        Example
        --------------
        >>>> a = AutoDiff.rAD(8.0)
        >>>> b = AutoDiff.rAD(8.0)
        >>>> c = AutoDiff.rAD(5.0)
        >>>> assert a != c
        >>>> assert (a == b) == False
        '''
        if self.val == other.val and self.der == other.der:
            return False
        else:
            return True

    def outer(self):
        '''
        Set gradient of outer function to 1.0. Must be called when function is defined.
        
        Returns
        --------------
        out: self.der = 1.0
        '''
        self.der = 1.0

def sin(x):
    '''
    sin(object)
    
    Return the sine of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the sine of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.sin(1.0)
        0.8414709848078965
    >>>> b = AutoDiff.rAD(8.0)
    >>>> c = AutoDiff.sin(b)
    >>>> c.get_val()
        0.9893582466233818
    >>>> x = AutoDiff.fAD(8.0)
    >>>> y = AutoDiff.sin(x)
    >>>> y.get_val()
        0.9893582466233818
    '''
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
    '''
    cos(object)
    
    Return the cosine of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the sine of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.cos(1.0)
        0.5403023058681398
    >>>> b = AutoDiff.rAD(8.0)
    >>>> c = AutoDiff.cos(b)
    >>>> c.get_val()
        -0.14550003380861354
    >>>> x = AutoDiff.fAD(8.0)
    >>>> y = AutoDiff.cos(x)
    >>>> y.get_val()
        -0.14550003380861354
    '''
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
    '''
    arcsin(object)
    
    Return the inverse sine of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the inverse sine of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.arcsin(1.0)
        1.5707963267948966
    >>>> b = AutoDiff.rAD(-0.50)
    >>>> c = AutoDiff.arcsin(b)
    >>>> c.get_val()
        -0.5235987755982988
    >>>> x = AutoDiff.fAD(-0.50)
    >>>> y = AutoDiff.arcsin(x)
    >>>> y.get_val()
        -0.5235987755982988
    '''
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
    '''
    arccos(object)
    
    Return the inverse cosine of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the inverse cosine of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.arccos(1.0)
        0.0
    >>>> b = AutoDiff.rAD(-0.50)
    >>>> c = AutoDiff.arccos(b)
    >>>> c.get_val()
        2.0943951023931957
    >>>> x = AutoDiff.fAD(-0.50)
    >>>> y = AutoDiff.arccos(x)
    >>>> y.get_val()
        2.0943951023931957
    '''
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
    '''
    arctan(object)
    
    Return the inverse tangent of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the inverse tangent of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.arctan(1.0)
        0.7853981633974483
    >>>> b = AutoDiff.rAD(-0.50)
    >>>> c = AutoDiff.arctan(b)
    >>>> c.get_val()
        -0.46364760900080615
    >>>> x = AutoDiff.fAD(-0.50)
    >>>> y = AutoDiff.arctan(x)
    >>>> y.get_val()
        -0.46364760900080615
    '''
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
    '''
    arctan(object)
    
    Return the hyperbolic sine of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the hyperbolic sine of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.sinh(1.0)
        1.1752011936438014
    >>>> b = AutoDiff.rAD(-0.50)
    >>>> c = AutoDiff.sinh(b)
    >>>> c.get_val()
        -0.5210953054937474
    >>>> x = AutoDiff.fAD(-0.50)
    >>>> y = AutoDiff.sinh(x)
    >>>> y.get_val()
        -0.5210953054937474
    '''
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
    '''
    exp(object)
    
    Return the exponential of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the exponential of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.exp(1.0)
        2.718281828459045
    >>>> b = AutoDiff.rAD(-0.50)
    >>>> c = AutoDiff.exp(b)
    >>>> c.get_val()
        0.6065306597126334
    >>>> x = AutoDiff.fAD(-0.50)
    >>>> y = AutoDiff.exp(x)
    >>>> y.get_val()
        0.6065306597126334
    '''
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
    '''
    log(object)
    
    Return the natural logarithm of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the natural logarithm of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.log(1.0)
        0.0
    >>>> b = AutoDiff.rAD(0.50)
    >>>> c = AutoDiff.log(b)
    >>>> c.get_val()
        -0.6931471805599453
    >>>> x = AutoDiff.fAD(0.50)
    >>>> y = AutoDiff.log(x)
    >>>> y.get_val()
        -0.6931471805599453
    '''
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
    '''
    tan(object)
    
    Return the tangent of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the tangent of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.tan(1.0)
        1.557407724654902
    >>>> b = AutoDiff.rAD(0.50)
    >>>> c = AutoDiff.tan(b)
    >>>> c.get_val()
        0.5463024898437905
    >>>> x = AutoDiff.fAD(0.50)
    >>>> y = AutoDiff.tan(x)
    >>>> y.get_val()
        0.5463024898437905
    '''
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
    '''
    cosh(object)
    
    Return the hyperbolic cosine of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the hyperbolic cosine of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.cosh(1.0)
        1.5430806348152437
    >>>> b = AutoDiff.rAD(0.50)
    >>>> c = AutoDiff.cosh(b)
    >>>> c.get_val()
        1.1276259652063807
    >>>> x = AutoDiff.fAD(0.50)
    >>>> y = AutoDiff.cosh(x)
    >>>> y.get_val()
        1.1276259652063807
    '''
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
    '''
    tanh(object)
    
    Return the hyperbolic tangent of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the hyperbolic tangent of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.tanh(1.0)
        0.7615941559557649
    >>>> b = AutoDiff.rAD(0.50)
    >>>> c = AutoDiff.tanh(b)
    >>>> c.get_val()
        0.46211715726000974
    >>>> x = AutoDiff.fAD(0.50)
    >>>> y = AutoDiff.tanh(x)
    >>>> y.get_val()
        0.46211715726000974
    '''
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
    '''
    sqrt(object)
    
    Return the non-negative square-root of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the non-negative square-root of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>>> AutoDiff.sqrt(4.0)
        2.0
    >>>> b =  AutoDiff.rAD(4.0)
    >>>> c = AutoDiff.sqrt(b)
    >>>> c.get_val()
        2.0
    >>>> x = AutoDiff.fAD(9.0)
    >>>> y = AutoDiff.sqrt(x)
    >>>> y.get_val()
        3.0
    '''
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
    '''
    mul_by_row(val, der)
    
    Allows multiplication of forward-mode autodiff object with 2-dimensional derivatives.

    Parameters
    --------------
    val: array_like.
        values of variables for differentiation

    der: array_like.
        partial derivatives of variables for differentiation
    '''
    if np.array(der).ndim <= 1:
        return val*der
    else:
        result = [val[i]*der[i] for i in range(len(val))]
        return np.array(result)

def reset_der(rADs):
    '''
    reset_der(rADs)
    
    Reset derivatives of reverse-mode autodiff objects

    Parameters
    --------------
    rADs: a single reverse autodiff object, or an array of reverse autodiff objects.
        Reverse-mode autodiff objects

    Examples
    --------------
    >>>> x = AutoDiff.rAD(8)
    >>>> z = x**2
    >>>> z.outer()
    >>>> x.grad()
    >>>> AutoDiff.reset_der(x)
    >>>> x.der
        None
    '''
    try:
        rADs.der = None
        rADs.children = []
    except AttributeError:
        for rAD in rADs:
            rAD.der = None
            rAD.children = []
