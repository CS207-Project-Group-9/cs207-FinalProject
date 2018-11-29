class rAD:
    def __init__(self, value):
        self.val = value
        self.children = [] # list of (w,a), s.t. rAD is built upon self
        self.der = None

    def grad(self):
        if self.der is None:
            self.der = sum(w*a.grad() for w,a in self.children)
        return self.der

    def __add__(self, other):
        # self + other = z
        # dz/dself = 1 * self.grad_value
        # dz/dother = 1 * self.grad_value
        ad = rAD(self.val + other.val)
        self.children.append((1.0, ad))
        other.children.append((1.0, ad))
        return ad

    def __mul__(self, other):
        ad = rAD(self.val * other.val)
        self.children.append((other.val, ad))
        other.children.append((self.val, ad))
        return ad

    def __neg__(self):
        return rAD(-self.val, -self.grad())

    def __abs__(self):
        return rAD(abs(self.val),(self.val/abs(self.val))*self.grad())

    def __str__(self):
        return "AutoDiff Objec, val: {0}, der: {1}".format(self.val, self.grad())
    
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
        x.children.append((np.exp(x.val)*x.grad(),ad))
        return ad
    except AttributeError:
        return np.exp(x)       

def log(x): 
    try:
        if x.val <= 0:
            raise ValueError("Cannot take log of negative value")
        else:
            ad = rAD(np.log(x.val))
            x.children.append((x.grad()/x.val,ad))
            return ad
    except AttributeError:
        if x <= 0:
            raise ValueError("Cannot take log of negative value")
        else:
            return np.log(x)


x = rAD(0.5)
y = rAD(4.2)
z = x * y + sin(x)
z.outer()
# z.der = 1.0 #the last (outermost) var



print('x: {}, {}, {}'.format(x.val, x.der, x.grad()))
print('y: {}, {}, {}'.format(y.val, y.der, y.grad()))
print('z: {}, {}, {}'.format(z.val, z.der, z.grad()))
print('x: ', str(x))
print('y: ', str(y))
print('z: ', str(z))



assert abs(z.val - 2.579425538604203) <= 1e-15
assert abs(x.grad() - (y.val + math.cos(x.val))) <= 1e-15
assert abs(y.grad() - x.val) <= 1e-15
