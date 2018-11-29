import math

class rAD:
    def __init__(self, value):
        self.val = value
        self.children = [] # list of (weight,var), s.t. var is built upon self
        self.der = None

    def grad(self):
        if self.der is None:
            self.der = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.der

    def __add__(self, other):
        # self + other = z
        # dz/dself = 1 * self.grad_value
        # dz/dother = 1 * self.grad_value
        z = rAD(self.val + other.val)
        self.children.append((1.0, z))
        other.children.append((1.0, z))
        return z

    def __mul__(self, other):
        z = rAD(self.val * other.val)
        self.children.append((other.val, z))
        other.children.append((self.val, z))
        return z

    def __neg__(self):
        return rAD(-self.val, -self.grad())

    def __abs__(self):
        return rAD(abs(self.val),(self.val/abs(self.val))*self.grad())

    def __str__(self):
        return "AutoDiff Objec, val: {0}, der: {1}".format(self.val, self.grad())

    def outer(self):
        self.der = 1.0


def sin(x):
    z = rAD(math.sin(x.val))
    x.children.append((math.cos(x.val), z))
    return z

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
