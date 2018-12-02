import numpy as np
import numbers
import math

class rAD:
    def __init__(self, vals):
        # check dimension of 'value'
        if np.array(vals).ndim > 1:
            raise ValueError('Input should be a scaler or a vector of numbers.')
        for i in np.array([vals]).reshape(-1):
            if not isinstance(i,numbers.Number):
                raise TypeError('Input should be a scaler or a vector of numbers.')
        self.val = vals
        self.children = []
        self.der = None

    def grad(self):
        if self.der is None:
            self.der = sum(w*a.grad() for w,a in self.children)
        return self.der

    def __add__(self, other):
        try:
            ad = rAD(self.val + other.val)
            self.children.append((1.0, ad))
            other.children.append((1.0, ad))
            return ad
        except AttributeError:
            ad = rAD(self.val + other)
            self.children.append((1.0, ad))
            return ad

    def __radd__(self, other):
        try:
            ad = rAD(self.val + other.val)
            self.children.append((1.0, ad))
            other.children.append((1.0, ad))
            return ad
        except AttributeError:
            ad = rAD(self.val + other)
            self.children.append((1.0, ad))
            return ad

    def __sub__(self, other):
        try:
            ad = rAD(self.val - other.val)
            self.children.append((1.0, ad))
            other.children.append((-1.0, ad))
            return ad
        except AttributeError:
            ad = rAD(self.val - other)
            self.children.append((1.0, ad))
            return ad

    def __rsub__(self, other):
        try:
            ad = rAD(other.val - self.val)
            self.children.append((-1.0, ad))
            other.children.append((1.0, ad))
            return ad
        except AttributeError:
            ad = rAD(other - self.val)
            self.children.append((-1.0, ad))
            return ad

    def __mul__(self, other):
        try:
            ad = rAD(self.val * other.val)
            self.children.append((other.val, ad))
            other.children.append((self.val, ad))
            return ad
        except AttributeError:
            ad = rAD(self.val * other)
            self.children.append((other, ad))
            return ad

    def __rmul__(self, other):
        try:
            ad = rAD(self.val * other.val)
            self.children.append((other.val, ad))
            other.children.append((self.val, ad))
            return ad
        except AttributeError:
            ad = rAD(self.val * other)
            self.children.append((other, ad))
            return ad

    def __truediv__(self, other):
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
        return "AutoDiff Object, val: {0}, der: {1}".format(self.val, self.grad())
    
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
    try:
        ad = rAD(np.sin(x.val))
        x.children.append((np.cos(x.val),ad))
        return ad
    except AttributeError:
        return np.sin(x)

def cos(x):
    try:
        ad = rAD(np.cos(x.val))
        x.children.append((-np.sin(x.val),ad))
        return ad
    except AttributeError:
        return np.cos(x)

def exp(x):
    try:
        ad = rAD(np.exp(x.val))
        x.children.append((np.exp(x.val),ad))
        return ad
    except AttributeError:
        return np.exp(x)       

def log(x): 
    try:
        if x.val <= 0:
            raise ValueError("Cannot take log of negative value")
        else:
            ad = rAD(np.log(x.val))
            x.children.append((1/x.val,ad))
            return ad
    except AttributeError:
        if x <= 0:
            raise ValueError("Cannot take log of negative value")
        else:
            return np.log(x)

def reset_der(rADs):
    try:
        rADs.der = None
        rADs.children = []
    except AttributeError:
        for rAD in rADs:
            rAD.der = None
            rAD.children = []

##if __name__ == "__main__":
##    x = rAD(0.5)
##    y = rAD(4.2)
##    z = x * y + sin(x)
##    z.outer()
##    # z.der = 1.0 #the last (outermost) var
##
##
##
##    print('x: {}, {}, {}'.format(x.val, x.der, x.grad()))
##    print('y: {}, {}, {}'.format(y.val, y.der, y.grad()))
##    print('z: {}, {}, {}'.format(z.val, z.der, z.grad()))
##    print('x: ', str(x))
##    print('y: ', str(y))
##    print('z: ', str(z))
##
##
##
##    assert abs(z.val - 2.579425538604203) <= 1e-15
##    assert abs(x.grad() - (y.val + math.cos(x.val))) <= 1e-15
##    assert abs(y.grad() - x.val) <= 1e-15
##
