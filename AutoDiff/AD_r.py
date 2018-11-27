import math

class Var:
    def __init__(self, value):
        self.value = value
        self.children = [] # list of (weight,var), s.t. var is built upon self
        self.grad_value = None

    def grad(self):
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad()
                                  for weight, var in self.children)
        return self.grad_value

    def __add__(self, other):
        # self + other = z
        # dz/dself = 1 * self.grad_value
        # dz/dother = 1 * self.grad_value
        z = Var(self.value + other.value)
        self.children.append((1.0, z))
        other.children.append((1.0, z))
        return z

    def __mul__(self, other):
        z = Var(self.value * other.value)
        self.children.append((other.value, z))
        other.children.append((self.value, z))
        return z

def sin(x):
    z = Var(math.sin(x.value))
    x.children.append((math.cos(x.value), z))
    return z

x = Var(0.5)
y = Var(4.2)
z = x * y + sin(x)
z.grad_value = 1.0 #the last (outermost) var



print('x: {}, {}, {}'.format(x.value, x.grad_value, x.grad()))
print('y: {}, {}, {}'.format(y.value, y.grad_value, y.grad()))
print('z: {}, {}, {}'.format(z.value, z.grad_value, z.grad()))

assert abs(z.value - 2.579425538604203) <= 1e-15
assert abs(x.grad() - (y.value + math.cos(x.value))) <= 1e-15
assert abs(y.grad() - x.value) <= 1e-15
