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
    >>> from Bambanta import AutoDiff
    >>> a = AutoDiff.create_f(3) 
    >>> a.val
    array([3])
    >>> a.der
    array([[1]])
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
    >>> from Bambanta import AutoDiff
    >>> a = fAD(5.0)
    >>> a.val
    array([ 5.])
    >>> a.der
    array([[1]])
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
        '''
        try: # assume other is of AutoDiff type
            return fAD(other.val/self.val, mul_by_row(1/self.val,other.der)-mul_by_row(other.val/(self.val**2),self.der))
        except AttributeError: # assume other is a number
            return fAD(other/self.val,mul_by_row(-other/(self.val**2),self.der))
            # if other is not a number, a TypeError will be raised

    def __pow__(self,exp):
        '''
        Support exponentiation of a forward autodiff object
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
        '''
        return fAD(-self.val, -self.der)

    def __abs__(self):
        '''
        Returns
        --------------
        out: the absolute of the autodiff object as a forward autodiff object
        '''
        return fAD(abs(self.val), mul_by_row(self.val/abs(self.val),self.der))

    def __repr__(self):
        '''
        Returns
        --------------
        out: 'fAD(values, derivatives)'
            outputs autodiff object values, and partial derivatives
        '''
        return "{0}({1},{2})".format(self.__class__.__name__, self.get_val(), self.get_jac())

    def __str__(self):
        '''
        Returns
        --------------
        out: "Forward-mode AutoDiff Object, value(s): values, partial derivative(s): derivatives" 
            outputs autodiff object values, and partial derivatives.
        '''
        return "Forward-mode AutoDiff Object, value(s): {0}, partial derivative(s): {1}".format(self.get_val(), self.get_jac())

    def __len__(self):
        '''
        Returns
        --------------
        out: number of variable values 
        '''
        return len(self.val)

    def __eq__(self, other):
        '''
        Allow comparisons between two equal forward autodiff objects
        '''
        if self.val==other.val and self.der==other.der:
            return True
        else:
            return False

    def __ne__(self, other):
        '''
        Allow comparisons between two unequal forward autodiff objects
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
        >>> from Bambanta import AutoDiff
        >>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>> f = 4*x + y
        >>> f.get_val()
        27.0

        multiple functions:
        >>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>> f1 = 4*x + y
        >>> f2 = x**3 - y
        >>> f = AutoDiff.stack_f([f1, f2])
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
        >>> from Bambanta import AutoDiff
        >>> x, y = AutoDiff.create_f([5.0, 7.0])
        >>> f = 4*x + y
        >>> f.get_jac()
        array([4, 1])
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
    >>> from Bambanta import AutoDiff
    >>> a = AutoDiff.create_r(2.0)
    >>> f = AutoDiff.sin(a)
    >>> f.outer()
    >>> f.get_val() #outputs function value
    0.90929742682568171
    >>> a.get_grad() #outputs df/da
    -0.41614683654714241
    '''
    if np.array(vals).ndim == 0:
        return rAD(vals)
    elif np.array(vals).ndim > 2:
        raise ValueError('Input is at most 2D.')
    else:
        ADs = [rAD(val) for val in vals]
        return ADs

def stack_r(vals, functions):
    '''
    stack_r(vals，functions)
    
    Initiate vector of functions for differentiation.
    
    Parameters
    --------------
    vals: array_like
        input reverse-mode autodiff variable values

    functions: array_like
        input functions for differentiation
        *functions must share an equal number of variables for differentiation*
        *functions must only contain computations supported by reverse-mode
        autodiff objects*
                
    Returns
    --------------
    out: Jacobian matrix of partial derivatives

    Examples
    --------------
    >>> from Bambanta import AutoDiff
    >>> def f1(x, y):
    ...  return 2*x + y
    >>> def f2(x, y):
    ...  return 3*x + 2*y
    >>> f = AutoDiff.stack_r([1, 3], [f1, f2])
    >>> f[0]
    array([5, 9])
    >>> f[1][0]
    array([ 2.,  1.])
    >>> f[1][1]
    array([ 3.,  2.])
    
    '''
    jac = []
    f_vals = []
    for f in functions:
        vars = [rAD(val) for val in vals]
        f_obj = f(*vars)
        f_obj.outer()
        f_vals.append(f_obj.get_val())
        grad = [var.get_grad() for var in vars]
        jac.append(grad)
    return np.array(f_vals), np.array(jac)

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
    >>> from Bambanta import AutoDiff
    >>> a = AutoDiff.rAD(5.0)
    >>> f = 2**a
    >>> f.outer()
    >>> f.get_val() #output function value
    32.0
    >>> a.get_grad() #output df/da
    22.180709777918249
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
        >>> from Bambanta import AutoDiff
        >>> a= AutoDiff.rAD([5.0])
        >>> f = 4*a
        >>> f.outer()
        >>> a.grad()
        array([ 4.])
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
        >>> from Bambanta import AutoDiff
        >>> x, y = AutoDiff.create_r([5.0, 7.0])
        >>> f = 4*x + y
        >>> f.outer()
        >>> f.get_val()
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
        >>> from Bambanta import AutoDiff
        >>> a,b = AutoDiff.create_r([[1,2],[3,4]])
        >>> f = 4*a + 3**b
        >>> f.outer()
        >>> a.get_grad()
        array([ 4.,  4.])
        >>> b.get_grad()
        array([ 29.66253179,  88.98759538])
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
        '''
        return self + other
        # try:
        #     ad = rAD(self.val + other.val)
        #     self.children.append((np.array([1.0]*len(self.val)), ad))
        #     other.children.append((np.array([1.0]*len(self.val)), ad))
        #     return ad
        # except AttributeError:
        #     ad = rAD(self.val + other)
        #     self.children.append((np.array([1.0]*len(self.val)), ad))
        #     return ad

    def __sub__(self, other):
        '''
        Support subtraction between:
        1. reverse autodiff objects
        2. a reverse autodiff object and a number
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
        '''
        return - self + other
        # try:
        #     ad = rAD(other.val - self.val)
        #     self.children.append((np.array([-1.0]*len(self.val)), ad))
        #     other.children.append((np.array([1.0]*len(self.val)), ad))
        #     return ad
        # except AttributeError:
        #     ad = rAD(other - self.val)
        #     self.children.append((np.array([-1.0]*len(self.val)), ad))
        #     return ad

    def __mul__(self, other):
        '''
        Support multiplication of:
        1. reverse autodiff objects
        2. a reverse autodiff object and a number
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
        '''
        return self * other
        # try:
        #     ad = rAD(self.val * other.val)
        #     self.children.append((other.val, ad))
        #     other.children.append((self.val, ad))
        #     return ad
        # except AttributeError:
        #     ad = rAD(self.val * other)
        #     self.children.append((np.array([other]*len(self.val)), ad))
        #     return ad

    def __truediv__(self, other):
        '''
        Support division between:
        1. reverse autodiff objects
        2. a reverse autodiff and a number
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
        '''
        return self**(-1) * other
        # try:
        #     ad = rAD(other.val / self.val)
        #     self.children.append((-other.val/(self.val**2), ad))
        #     other.children.append((1/self.val, ad))
        #     return ad
        # except AttributeError:
        #     ad = rAD(other / self.val)
        #     self.children.append((-other/(self.val**2), ad))
        #     return ad

    def __pow__(self, other):
        '''
        Support exponentiation of a reverse autodiff object
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
        '''
        new = rAD(-self.val)
        self.children.append((np.array([-1.0]*len(self.val)), new))
        return new

    def __abs__(self):
        '''
        Returns
        --------------
        out: the absolute of the autodiff object as a reverse autodiff object       
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
        '''
        return "Reverse AutoDiff Object, value(s): {0}, gradient: {1}".format(self.val, self.grad())

    def __eq__(self, other):
        '''
        Allow comparisons between two reverse autodiff objects
        '''
        if self.val == other.val and self.der == other.der:
            return True
        else:
            return False

    def __ne__(self, other):
        '''
        Allow comparisons between two reverse autodiff objects
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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.sin(1.0)
    0.8414709848078965
    >>> b = AutoDiff.rAD(8.0)
    >>> c = AutoDiff.sin(b)
    >>> c.get_val()
    0.98935824662338179
    >>> x = AutoDiff.fAD(8.0)
    >>> y = AutoDiff.sin(x)
    >>> y.get_val()
    0.98935824662338179
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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.cos(1.0)
    0.54030230586813977
    >>> b = AutoDiff.rAD(8.0)
    >>> c = AutoDiff.cos(b)
    >>> c.get_val()
    -0.14550003380861354
    >>> x = AutoDiff.fAD(8.0)
    >>> y = AutoDiff.cos(x)
    >>> y.get_val()
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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.arcsin(1.0)
    1.5707963267948966
    >>> b = AutoDiff.rAD(-0.50)
    >>> c = AutoDiff.arcsin(b)
    >>> c.get_val()
    -0.52359877559829893
    >>> x = AutoDiff.fAD(-0.50)
    >>> y = AutoDiff.arcsin(x)
    >>> y.get_val()
    -0.52359877559829893
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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.arccos(1.0)
    0.0
    >>> b = AutoDiff.rAD(-0.50)
    >>> c = AutoDiff.arccos(b)
    >>> c.get_val()
    2.0943951023931957
    >>> x = AutoDiff.fAD(-0.50)
    >>> y = AutoDiff.arccos(x)
    >>> y.get_val()
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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.arctan(1.0)
    0.78539816339744828
    >>> b = AutoDiff.rAD(1.0)
    >>> c = AutoDiff.arctan(b)
    >>> c.get_val()
    0.78539816339744828
    >>> x = AutoDiff.fAD(1.0)
    >>> y = AutoDiff.arctan(x)
    >>> y.get_val()
    0.78539816339744828
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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.sinh(1.0)
    1.1752011936438014
    >>> b = AutoDiff.rAD(-0.50)
    >>> c = AutoDiff.sinh(b)
    >>> c.get_val()
    -0.52109530549374738
    >>> x = AutoDiff.fAD(-0.50)
    >>> y = AutoDiff.sinh(x)
    >>> y.get_val()
    -0.52109530549374738
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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.exp(1.0)
    2.7182818284590451
    >>> b = AutoDiff.rAD(1.0)
    >>> c = AutoDiff.exp(b)
    >>> c.get_val()
    2.7182818284590451
    >>> x = AutoDiff.fAD(1.0)
    >>> y = AutoDiff.exp(x)
    >>> y.get_val()
    2.7182818284590451
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

def logistic(x):
    '''
    logistic(object)
    
    Return the standard logistic of the input object.

    Parameters
    --------------
    object: a number, or an autodiff object, whether forward-, or reverse-mode.  

    Returns
    --------------
    out: the standard logistic of the input object.
        Numeric if input is a number, or an autodiff object if input is an autodiff object.

    Example
    --------------
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.logistic(1.0)
    0.7310585786300049
    >>> b = AutoDiff.rAD(1.0)
    >>> c = AutoDiff.logistic(b)
    >>> c.get_val()
    0.7310585786300049
    >>> x = AutoDiff.fAD(1.0)
    >>> y = AutoDiff.logistic(x)
    >>> y.get_val()
    0.7310585786300049
    '''
    try:  # x <- rAD
        ad = rAD(1/(1+np.exp(-x.val)))
        x.children.append((np.exp(-x.val)/((np.exp(-x.val)+1)**2),ad))
        return ad
    except AttributeError: 
        try: # x <- fAD
            return fAD(1/(1+np.exp(-x.val)), 
                mul_by_row(np.exp(-x.val)/((np.exp(-x.val)+1)**2),x.der))
        except AttributeError: # x <- numeric
            return 1/(1+np.exp(-x))

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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.log(1.0)
    0.0
    >>> b = AutoDiff.rAD(1.0)
    >>> c = AutoDiff.log(b)
    >>> c.get_val()
    0.0
    >>> x = AutoDiff.fAD(1.0)
    >>> y = AutoDiff.log(x)
    >>> y.get_val()
    0.0
    '''
    try: # x <- rAD
        ad = rAD(np.log(x.val)/np.log(base))
        x.children.append((1/(x.val*np.log(base)),ad))
        return ad
    except AttributeError:
        try: # x <- fAD
            return fAD(np.log(x.val)/np.log(base), mul_by_row(1/(x.val*np.log(base)),x.der))
        except AttributeError: # x <- numeric
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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.tan(1.0)
    1.5574077246549023
    >>> b = AutoDiff.rAD(1.0)
    >>> c = AutoDiff.tan(b)
    >>> c.get_val()
    1.5574077246549023
    >>> x = AutoDiff.fAD(1.0)
    >>> y = AutoDiff.tan(x)
    >>> y.get_val()
    1.5574077246549023
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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.cosh(1.0)
    1.5430806348152437
    >>> b = AutoDiff.rAD(0.50)
    >>> c = AutoDiff.cosh(b)
    >>> c.get_val()
    1.1276259652063807
    >>> x = AutoDiff.fAD(0.50)
    >>> y = AutoDiff.cosh(x)
    >>> y.get_val()
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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.tanh(1.0)
    0.76159415595576485
    >>> b = AutoDiff.rAD(0.50)
    >>> c = AutoDiff.tanh(b)
    >>> c.get_val()
    0.46211715726000974
    >>> x = AutoDiff.fAD(0.50)
    >>> y = AutoDiff.tanh(x)
    >>> y.get_val()
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
    >>> from Bambanta import AutoDiff
    >>> AutoDiff.sqrt(4.0)
    2.0
    >>> b =  AutoDiff.rAD(4.0)
    >>> c = AutoDiff.sqrt(b)
    >>> c.get_val()
    2.0
    >>> x = AutoDiff.fAD(9.0)
    >>> y = AutoDiff.sqrt(x)
    >>> y.get_val()
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
    >>> from Bambanta import AutoDiff
    >>> x = AutoDiff.rAD(8)
    >>> z = x**2
    >>> z.outer()
    >>> x.grad()
    array([ 16.])
    >>> AutoDiff.reset_der(x)
    >>> x.der
    '''
    try:
        rADs.der = None
        rADs.children = []
    except AttributeError:
        for rAD in rADs:
            rAD.der = None
            rAD.children = []

# if __name__ == '__main__':
#     import doctest
#     doctest.testmod()
