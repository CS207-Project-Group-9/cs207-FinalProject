#test_AutoDiff.py
#Dec. 2, 2018

#This test suite is associated with file 'AutoDiff.py', which
#implements forward-mode and reverse-mode automatic differentiation.

#import unit testing packages pytest and numpy testing
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_approx_equal

from Bambanta import AutoDiff

#AD_create_f allows for simultaneous assignment 
#of AD instances
def test_AD_create_f():
    a = AutoDiff.create_f(3)
    assert a.val == [3], a.der == [[1]]
    a, b, c = AutoDiff.create_f([1, 2, 3])
    assert a.val == [1], a.der == [[1,0,0]]
    assert b.val == [2], b.der == [[0,1,0]]
    assert c.val == [3], c.der == [[0,0,1]]
    a, b = AutoDiff.create_f([[1, 2],[3, 4]])
    assert_array_equal(a.val, np.array([1,2]))
    assert_array_equal(a.der, np.array([[1,0],[1,0]]))
    assert_array_equal(b.val, np.array([3,4]))
    assert_array_equal(b.der, np.array([[0,1],[0,1]]))
    with pytest.raises(ValueError):
        AutoDiff.create_f([[[1,2],[1,2]], [[3,4],[3,4]]])

#AD_stack takes in multiple AD instances
#in the form of numpy arrays, returns 
#values as a vector and derivatives as a matrix
def test_AD_stack_f():
    a, b, c = AutoDiff.create_f([1, 2, 3])
    c = AutoDiff.stack_f([a, b, c])
    assert_array_equal(c.val, np.array([1,2,3]))
    assert_array_equal(c.der, np.array([[1,0,0],[0,1,0],[0,0,1]]))

#Test whether constructor of AutoDiff class 
#returns proper values, derivatives, and errors
def test_fAD_constructor_init():
    #test single value input
    a = 5.0
    b = AutoDiff.fAD(a,1)
    #test multiple value input
    c = AutoDiff.fAD([1,5], [[1,0],[0,1]])
    assert_array_equal(b.val, np.array([5.0]))
    assert_array_equal(b.der, np.array([[1]]))
    assert_array_equal(b.get_val(), np.array([5.0])) #test get_val()
    assert_array_equal(b.get_jac(), np.array([[1]])) #test get_der()
    assert_array_equal(c.val, np.array([1, 5]))
    assert_array_equal(c.der, np.array([[1, 0], [0, 1]]))
    assert_array_equal(c.get_val(), np.array([1, 5])) #test get_val()
    assert_array_equal(c.get_jac(), np.array([[1, 0], [0, 1]])) #test get_der()
    #inputs ought not to be type other than scaler, list or array of numbers
    with pytest.raises(TypeError):
        AutoDiff.fAD('hello','friend')
    with pytest.raises(TypeError):
        AutoDiff.fAD([5.0], 'test')
    #check maximal dimensions of val and der
    with pytest.raises(ValueError):
        AutoDiff.fAD([[1],[2]], [[1,0],[0,1]])
    with pytest.raises(ValueError):
        AutoDiff.fAD([[5.0]], [[[1,0,0],[0,1,0]]])
    #check if dimension of derivative input matches that of value input
    with pytest.raises(ValueError):
        AutoDiff.fAD([1], [[1],[2]])
    with pytest.raises(ValueError):
        AutoDiff.fAD([1,2], [1,2,3,4])
    with pytest.raises(ValueError):
        AutoDiff.fAD([1,2,3],[[1,2,3],[1,2]])
    with pytest.raises(ValueError):
        AutoDiff.fAD([1,2,3],[[1,0,0],[0,1,0]])
    with pytest.raises(ValueError):
        AutoDiff.fAD([])
    #check variable type
    with pytest.raises(TypeError):
        AutoDiff.fAD([1,2], [['a','b','c'],['d','e','f']])
    with pytest.raises(TypeError):
        AutoDiff.fAD(['a'], [[1,2]])

        
#Test whether addition works between AD instances, 
#and between AD instance and number, regardless of order
def test_fAD_add():
    x, y = AutoDiff.create_f([5.0, 7.0])
    z = 3.0
    sum1 = x + y #AD+AD
    sum2 = x + z #AD+number
    sum3 = z + y #test __radd__: number+AD
    assert sum1.val == [12.0]
    assert_array_equal(sum1.der, np.array([[1, 1]]))
    assert sum2.val == [8.0]
    assert_array_equal(sum2.der, np.array([[1, 0]]))
    assert sum3.val == [10.0]
    assert_array_equal(sum3.der, np.array([[0, 1]]))
    with pytest.raises(TypeError):
        x + 'hello'
    with pytest.raises(TypeError):
        'friend' + y

#Test whether subtraction works between AD instances,
#and between AD instance and number, regardless of order
def test_fAD_sub():
    x, y = AutoDiff.create_f([5.0, 7.0])
    z = 3.0
    m = 10.0
    sub1 = y - x #AD-AD
    sub2 = x - z #AD-number
    sub3 = m - x #test __rsub__: number-AD
    assert sub1.val == [2.0]
    assert_array_equal(sub1.der, np.array([[-1, 1]]))
    assert sub2.val == [2.0]
    assert_array_equal(sub2.der, np.array([[1, 0]]))
    assert sub3.val == [5.0]
    assert_array_equal(sub3.der, np.array([[-1, 0]]))
    with pytest.raises(TypeError):
        x - 'hello'
    with pytest.raises(TypeError):
        'friend' - y

#Test whether multiplication works between AD instances,
#and between AD instance and number, regardless of order
def test_fAD_mul():
    x, y = AutoDiff.create_f([5.0, 7.0])
    z = 3.0
    mul1 = x * y #AD*AD
    mul2 = x * z #AD*number
    mul3 = z * y #test __rmul__: number*AD
    assert mul1.val == [35.0]
    assert_array_equal(mul1.der, np.array([[7.0, 5.0]]))
    assert mul2.val == [15.0]
    assert_array_equal(mul2.der, np.array([[3.0, 0]]))
    assert mul3.val == [21.0]
    assert_array_equal(mul3.der, np.array([[0 , 3.0]]))
    with pytest.raises(TypeError):
        x * 'hello'
    with pytest.raises(TypeError):  
        'friend' * y
    
#Test whether division works between AD instances,
#and between AD instance and number, regardless of order
def test_fAD_div():
    x, y = AutoDiff.create_f([4.0, 8.0])
    z = 2.0
    div1 = y / x #AD/AD
    div2 = x / z #AD/number
    div3 = z / y #test __rtruediv__: number/AD
    assert div1.val == [2.0]
    assert_array_equal(div1.der, np.array([[-0.5, 0.25]]))
    assert div2.val == [2.0]
    assert_array_equal(div2.der, np.array([[0.5, 0]]))
    assert div3.val == [0.25]
    assert_array_equal(div3.der, np.array([[0, -0.03125]]))
    with pytest.raises(TypeError):
        x / 'hello'
    with pytest.raises(TypeError):
        'friend' / y
    
#Test whether differetiation with power works when
#AD instance is the base, and when AD instance is 
#the exponent
def test_fAD_pow():
    x, y = AutoDiff.create_f([2.0, 3.0])
    a, b = AutoDiff.create_f(np.array([1.0, 2.0]))
    z = 5.0
    power1 = (x*y) ** z #AD**number
    power2 = z ** (a*b) #test __rpow__: number**AD
    power3 = x**y
    assert power1.val == [7776.0]
    assert_array_equal(power1.der, np.array([[19440., 12960.]]))
    assert power2.val == [25.0]
    assert_array_almost_equal(power2.der, np.array([[80.47189562, 40.23594781]]))
    assert power3.val == [8.0]
    assert_array_almost_equal(power3.der, np.array([[12., 5.54517744]]))
    with pytest.raises(TypeError):
        x ** 'hello'
    with pytest.raises(TypeError):
        'friend' ** y

#Test whether taking the negative of AD instance works
def test_fAD_neg():
    x, y = AutoDiff.create_f([2.0, 8.0])
    neg1 = -x
    neg2 = -(x/y)
    assert neg1.val == [-2.0]
    assert_array_equal(neg1.der, np.array([[-1, 0]]))
    assert neg2.val == [-0.25]
    assert_array_equal(neg2.der, np.array([[-0.125, 0.03125]]))

#Test __abs__
def test_fAD_abs():
    a = AutoDiff.fAD(-8,1)
    b = abs(a)
    assert b.val == [8]
    assert b.der == [[-1.]]
        
#Test __str__ and __repr__
def test_fAD_print():
    a, b = AutoDiff.create_f([5.0, 7.0])
    f = 4*a + b
    assert f.get_val() == 27.0
    assert_array_equal(f.get_jac(), np.array([4, 1]))
    assert 'Forward-mode AutoDiff Object' in str(f)
    assert 'value(s)' in str(f)
    assert 'partial derivative(s)' in str(f)
    assert 'fAD' in repr(f)

#Test __len__
def test_fAD_len():
    a, b = AutoDiff.create_f([2.0, 8.0])
    c = AutoDiff.stack_f([a,b])
    assert len(c) == 2

#Test __eq__
def test_fAD_eq():
    a = AutoDiff.fAD(8.0,1)
    b = AutoDiff.fAD(8.0,1)
    c = AutoDiff.fAD(5.0,1)
    assert a == b
    assert (a == c) == False

#Test __ne__
def test_fAD_ne():
    a = AutoDiff.fAD(8.0,1)
    b = AutoDiff.fAD(8.0,1)
    c = AutoDiff.fAD(5.0,1)
    assert a != c
    assert (a != b) == False


#Test create_r()
def test_rAD_create_r():
    a, b, c = AutoDiff.create_r([1,2,3])
    f1 = 2*a + b**3 +AutoDiff.cos(c)
    f1.outer()
    assert a.get_grad() == 2.0
    assert b.get_grad() == 12.0
    assert_array_almost_equal(np.array(c.get_grad()), np.array(-0.1411200080598672))
    assert_array_almost_equal(np.array(f1.get_val()), np.array(9.010007503399555))
    #check dimension
    with pytest.raises(ValueError):
        AutoDiff.create_r([[[1]],[[2]]])

#Test stack_r()
def test_rAD_stack_r():
    def f1(x, y):
        return 2*x + y
    def f2(x, y):
        return 3*x + 2*y
    v,j = AutoDiff.stack_r([1, 3], [f1, f2])
    assert_array_equal(v, np.array([5.0, 9.0]))
    assert_array_equal(j[0], np.array([2.0, 1.0]))
    assert_array_equal(j[1], np.array([3.0, 2.0]))
    
    
#Test whether constructor of rAD class returns proper
#values, children, derivatives, and errors
def test_rAD_constructor_init():
    a = AutoDiff.rAD(8)
    assert a.val == 8
    assert a.children == []
    assert a.der == None
    #inputs ought not to be type other than scaler, list or array of numbers
    with pytest.raises(TypeError):
        AutoDiff.rAD('test')
    #inputs ought not be matrix or of higher order dimension
    with pytest.raises(ValueError):
        AutoDiff.rAD([[1,2],[3,4]])

#Test whether the reset_der function correctly resets the 
#derivatives and children of the given rAD objects
def test_rAD_reset_der():
    #reset_der works on single rAD object
    x = AutoDiff.rAD(8)
    z = x ** 2
    z.outer()
    x.grad()
    AutoDiff.reset_der(x)
    assert x.children == []
    assert x.der == None
    #reset_der works on a list/array of rAD objects
    x, y = AutoDiff.rAD(8), AutoDiff.rAD(5)
    z = x * y
    z.outer()
    x.grad()
    y.grad()
    AutoDiff.reset_der([x,y])
    assert x.children == []
    assert x.der == None
    assert y.children == []
    assert y.der == None

#Test whether addition works between rAD instances, 
#and between rAD instance and number, regardless of order
def test_rAD_add():
    x, y, z = AutoDiff.rAD(5.0), AutoDiff.rAD(7.0), 3.0
    #rAD+rAD
    sum1 = x + y
    sum1.outer()
    assert sum1.get_val() == 12.0
    assert x.grad() == 1
    assert y.grad() == 1
    AutoDiff.reset_der([x,y])
    #rAD+number
    sum2 = x + z
    sum2.outer()
    assert sum2.get_val() == 8.0
    assert x.grad() == 1
    #number+rAD <- test __radd__
    sum3 = z + y
    sum3.outer()
    assert sum3.get_val() == 10.0
    assert y.grad() == 1
    #rAD cannot be added to non-rAD and non-numeric types
    with pytest.raises(TypeError):
        x + 'hello'
    with pytest.raises(TypeError):
        'friend' + y

#Test whether subtraction works between rAD instances, 
#and between rAD instance and number, regardless of order
def test_rAD_sub():
    x, y, z = AutoDiff.rAD(5.0), AutoDiff.rAD(7.0), 3.0
    #rAD-rAD
    sub1 = y - x
    sub1.outer()
    assert sub1.get_val() == 2.0
    assert x.grad() == -1
    assert y.grad() == 1
    AutoDiff.reset_der([x,y])
    #rAD+number
    sub2 = x - z
    sub2.outer()
    assert sub2.get_val() == 2.0
    assert x.grad() == 1
    #number+rAD <- test __rsub__
    sub3 = z - y
    sub3.outer()
    assert sub3.get_val() == -4.0
    assert y.grad() == -1
    #rAD cannot subtract or be subtracted by non-rAD and non-numeric types
    with pytest.raises(TypeError):
        x - 'hello'
    with pytest.raises(TypeError):
        'friend' - y

#Test whether multiplication works between AD instances,
#and between AD instance and number, regardless of order
def test_rAD_mul():
    x, y, z = AutoDiff.rAD(5.0), AutoDiff.rAD(7.0), 3.0
    #rAD*rAD
    mul1 = x * y
    mul1.outer()
    assert mul1.val == 35.0
    assert x.grad() == 7.0
    assert y.grad() == 5.0
    AutoDiff.reset_der([x,y])
    #rAD*number
    mul2 = x * z
    mul2.outer()
    assert mul2.val == 15.0
    assert x.grad() == 3.0
    #number*rAD <- test __rmul__
    mul3 = z * y
    mul3.outer()
    assert mul3.val == 21.0
    assert y.grad() == 3.0
    #rAD cannot be multiplied with non-rAD and non-numeric types
    with pytest.raises(TypeError):
        x * 'hello'
    with pytest.raises(TypeError):  
        'friend' * y

#Test whether division works between rAD instances, 
#and between rAD instance and number, regardless of order
def test_rAD_div():
    x, y, z = AutoDiff.rAD(4.0), AutoDiff.rAD(8.0), 2.0
    #rAD-rAD
    div1 = y / x
    div1.outer()
    assert div1.val == 2.0
    assert x.grad() == -0.5
    assert y.grad() == 0.25
    AutoDiff.reset_der([x,y])
    #rAD+number
    div2 = x / z
    div2.outer()
    assert div2.val == 2.0
    assert x.grad() == 0.5
    #number+rAD <- test __rsub__
    div3 = z / y
    div3.outer()
    assert div3.val == 0.25
    assert y.grad() == -0.03125
    #rAD cannot subtract or be subtracted by non-rAD and non-numeric types
    with pytest.raises(TypeError):
        x / 'hello'
    with pytest.raises(TypeError):
        'friend' / y

#Test whether power works between rAD instances, 
#and between rAD instance and number, regardless of order
def test_rAD_pow():
    x, y, z = AutoDiff.rAD(2.0), AutoDiff.rAD(3.0), 5.0
    a, b = AutoDiff.rAD(1.0), AutoDiff.rAD(2.0)
    #rAD**rAD
    pow1 = x ** y
    pow1.outer()
    assert pow1.val == 8.0
    assert_array_almost_equal(np.array([x.grad(), y.grad()]),
        np.array([[12.], [5.54517744]]))
    AutoDiff.reset_der([x,y])
    #rAD**number
    pow2 = (x * y) ** z
    pow2.outer()
    assert pow2.val == 7776.0
    assert_array_almost_equal(np.array([x.grad(), y.grad()]),
        np.array([[19440.], [12960.]]))
    #number**rAD <- test __rpow__
    pow3 = z ** (a * b)
    pow3.outer()
    assert pow3.val == 25.0
    assert_array_almost_equal(np.array([a.grad(), b.grad()]),
        np.array([[80.47189562], [40.23594781]]))
    #user cannot compute power between rAD with non-rAD and non-numeric types
    with pytest.raises(TypeError):
        x / 'hello'
    with pytest.raises(TypeError):
        'friend' / y

#Test whether taking the negative of rAD instance works
def test_rAD_neg():
    x = AutoDiff.rAD(6.5)
    y = AutoDiff.rAD(3.0)
    z = -x - AutoDiff.cos(y)
    z.outer()
    assert x.grad() == -1.0 
    assert_array_almost_equal(np.array(y.grad()), np.array(0.1411200080598672)) 
    assert_array_almost_equal(np.array(z.val), np.array(-5.510007503399555))

#Test rAD abs()
def test_rAD_abs():
    x = AutoDiff.rAD(-6.5)
    y = AutoDiff.rAD(3.0)
    z = abs(x)*AutoDiff.sin(y)
    z.outer()
    assert_array_almost_equal(np.array(z.val), np.array(0.9172800523891369))
    assert_array_almost_equal(np.array(x.grad()), np.array(-0.1411200080598672))
    assert_array_almost_equal(np.array(y.grad()), np.array(-6.4349512279028955))
    
#Test str() of rAD
def test_rAD_str():
    x = AutoDiff.rAD(-6.5)
    assert str(x) == 'Reverse AutoDiff Object, value(s): [-6.5], gradient: 0'

#Test __eq__
def test_rAD_eq():
    a = AutoDiff.rAD(8.0)
    b = AutoDiff.rAD(8.0)
    c = AutoDiff.rAD(5.0)
    assert a == b
    assert (a == c) == False

#Test __ne__
def test_rAD_ne():
    a = AutoDiff.rAD(8.0)
    b = AutoDiff.rAD(8.0)
    c = AutoDiff.rAD(5.0)
    assert a != c
    assert (a != b) == False

#Test whether taking the sine of AD instance returns the correct value
#Test whether the sin() function also apply to integers
def test_combined_sin():
    # fAD
    x = AutoDiff.fAD(1.0, [1, 0])
    y = AutoDiff.sin(x)
    assert_array_almost_equal(y.val, np.array([0.84147098]), decimal = 6)
    assert_array_almost_equal(y.der, np.array([[0.54030231, 0.]]), decimal = 6)
    # numeric
    a = 6.0
    b = AutoDiff.sin(a)
    assert b == -0.27941549819892586
    # rAD

#Test whether taking the cosine of AD instance returns the correct value
def test_combined_cos():
    # fAD
    a, b = AutoDiff.create_f([2.0, 8.0])
    c = AutoDiff.cos(a*b)
    assert_array_almost_equal(c.val, np.array([-0.95765948]), decimal = 6)
    assert_array_almost_equal(c.der, np.array([[2.30322653, 0.57580663]]), decimal = 6)
    # numeric
    x = 5.0
    y = AutoDiff.cos(x)
    assert y == pytest.approx(0.2836621854632263)

#Test whether taking the tangent of AD instance returns the correct value
def test_combined_tan():
    x = AutoDiff.fAD(5)
    y = AutoDiff.tan(x**2)    
    a = AutoDiff.rAD(5)
    b = AutoDiff.tan(a**2)
    b.outer()
    a.grad()
    num = 10/np.cos(25)**2
    assert(y.val[0] == np.tan(25))
    assert_approx_equal(y.der[0][0], num)
    assert(b.val == np.tan(25))
    assert_array_almost_equal(a.der,np.array(num))  
    assert(AutoDiff.tan(3) == np.tan(3))

#Test inverse sine
def test_combined_arcsin():
    #test fAD
    x, y = AutoDiff.create_f([0.25, -0.10])
    f = AutoDiff.arcsin(x) + AutoDiff.arcsin(y)
    #test numeric
    z = 1.0
    #test rAD
    a = AutoDiff.rAD(0.25)
    b = AutoDiff.rAD(-0.10)
    c = AutoDiff.arcsin(a) + AutoDiff.arcsin(b)
    c.outer()
    assert_array_almost_equal(f.val, np.array(0.15251283))
    assert_array_almost_equal(f.val, np.array(c.val))
    assert_array_almost_equal(f.der[0][0], np.array(1.03279556))
    assert_array_almost_equal(f.der[0][0], np.array(a.grad()))
    assert_array_almost_equal(f.der[0][1], np.array(b.grad()))
    assert_array_almost_equal(np.array(AutoDiff.arcsin(z)), np.array(1.5707963267948966))

#Test inverse cosine
def test_combined_arccos():
    #test fAD
    x, y = AutoDiff.create_f([0.40, -0.55])
    f = AutoDiff.arccos(x) + AutoDiff.arccos(y)
    #test numeric
    z = 0.5
    #test rAD
    a = AutoDiff.rAD(0.40)
    b = AutoDiff.rAD(-0.55)
    c = AutoDiff.arccos(a) + AutoDiff.arccos(b)
    c.outer()
    assert_array_almost_equal(f.val, np.array(3.31244005))
    assert_array_almost_equal(f.val, np.array(c.val))
    assert_array_almost_equal(f.der[0][0], np.array(-1.09108945))
    assert_array_almost_equal(f.der[0][0], np.array(a.grad()))
    assert_array_almost_equal(f.der[0][1], np.array(b.grad()))
    assert_array_almost_equal(np.array(AutoDiff.arccos(z)), np.array(1.0471975511965976))

#Test inverse tangent
def test_combined_arctan():
    #test fAD
    x, y = AutoDiff.create_f([0.30, -0.25])
    f = AutoDiff.arctan(x) + y
    #test numeric
    z = 0.1
    #test rAD
    a = AutoDiff.rAD(0.30)
    b = AutoDiff.rAD(-0.25)
    c = AutoDiff.arctan(a) + b
    c.outer()
    assert_array_almost_equal(f.val, np.array(0.04145679))
    assert_array_almost_equal(f.val, np.array(c.val), decimal = 4)
    assert_array_almost_equal(f.der[0][0], np.array(0.91743119))
    assert_array_almost_equal(f.der[0][0], np.array(a.grad()))
    assert_array_almost_equal(f.der[0][1], np.array(b.grad()))
    assert_array_almost_equal(np.array(AutoDiff.arctan(z)), np.array(0.09966865249116204))

#Test hyperbolic sine
def test_combined_sinh():
    #test fAD
    x, y = AutoDiff.create_f([5, -8.5])
    f = AutoDiff.sinh(x) + y
    #test numeric
    z = 2
    #test rAD
    a = AutoDiff.rAD(5)
    b = AutoDiff.rAD(-8.5)
    c = AutoDiff.sinh(a) + b
    c.outer()
    assert_array_almost_equal(f.val, np.array(65.70321058))
    assert_array_almost_equal(f.val, np.array(c.val), decimal = 4)
    assert_array_almost_equal(f.der[0][0], np.array(74.20994852))
    assert_array_almost_equal(f.der[0][0], np.array(a.grad()))
    assert_array_almost_equal(f.der[0][1], np.array(b.grad()))
    assert_array_almost_equal(np.array(AutoDiff.sinh(z)), np.array(3.6268604078470186))

def test_combined_cosh():
    x = AutoDiff.fAD(8)
    y = AutoDiff.cosh(x) + x
    a = AutoDiff.rAD(8)
    b = AutoDiff.cosh(a) - a
    b.outer()
    a.grad()
    assert y.val == np.cosh(8)+8
    assert y.der == np.sinh(8)+1
    assert b.val == np.cosh(8)-8
    assert a.der == np.sinh(8)-1
    assert AutoDiff.cosh(8) == np.cosh(8)

def test_combined_tanh():
    x = AutoDiff.fAD(3)
    y = AutoDiff.tanh(x**2)+x**2
    a = AutoDiff.rAD(3)
    b = AutoDiff.tanh(a**2)+a**2
    b.outer()
    a.grad()
    assert y.val == np.tanh(9)+9
    assert y.der == 6/(np.cosh(9)**2)+6
    assert b.val == np.tanh(9)+9
    assert a.der == 6/(np.cosh(9)**2)+6
    assert AutoDiff.tanh(3) == np.tanh(3)
    
def test_combined_sqrt():    
    x = AutoDiff.fAD(6)
    y = 4*AutoDiff.sqrt(x)
    a = AutoDiff.rAD(6)
    b = 4*AutoDiff.sqrt(a)
    b.outer()
    a.grad()
    assert y.val == 4*6**0.5
    assert_array_almost_equal(y.der,2*6**-0.5)
    assert b.val == 4*6**0.5
    assert_array_almost_equal(a.der,2*6**-0.5)
    assert AutoDiff.sqrt(5) == np.sqrt(5)

#Test whether taking the natural logarithm of AD instance returns the correct value
def test_combined_log():
    # fAD
    a, b = AutoDiff.create_f([-4.0, 8.0])
    new_base = 10
    assert_array_almost_equal(AutoDiff.log(b).val, np.array([2.07944154]), decimal = 6)
    assert_array_equal(AutoDiff.log(b).der, np.array([[0, 0.125]]))
    assert_array_almost_equal(AutoDiff.log(b,new_base).val, np.array([0.90308998]), decimal = 6)
    assert_array_almost_equal(AutoDiff.log(b,new_base).der, np.array([[0, 0.05428681]]), decimal = 6)
    # numeric
    x = 5.0
    assert AutoDiff.log(x) == 1.6094379124341003
    # rAD
    x = AutoDiff.rAD([2,3])
    z = AutoDiff.log(x,2)
    z.outer()
    assert_array_almost_equal(z.get_val(), np.array([1. , 1.5849625]))
    assert_array_almost_equal(x.get_grad(), np.array([0.72134752, 0.48089835]))

#Test exp()
def test_combined_exp():
    # fAD
    x,y = AutoDiff.create_f([2, 3])
    z = AutoDiff.exp(x)
    assert_array_almost_equal(z.val, np.array([7.3890561]))
    assert_array_almost_equal(z.der, np.array([[7.3890561, 0. ]]))
    # numeric
    a = 5.0
    b = AutoDiff.exp(a)
    assert b == 148.4131591025766
    # rAD
    x = AutoDiff.rAD([2, 3])
    z = AutoDiff.exp(x)
    z.outer()
    assert_array_almost_equal(z.get_val(), np.array([7.3890561 , 20.08553692]))
    assert_array_almost_equal(x.get_grad(), np.array([7.3890561 , 20.08553692]))

#Test exp()
def test_combined_logistic():
    # fAD
    x,y = AutoDiff.create_f([2, 3])
    z = AutoDiff.logistic(x)
    assert_array_almost_equal(z.val, np.array([0.88079708]))
    assert_array_almost_equal(z.der, np.array([[0.10499359, 0. ]]))
    # numeric
    a = 5.0
    b = AutoDiff.logistic(a)
    assert b == 0.9933071490757153
    # rAD
    x = AutoDiff.rAD([2, 3])
    z = AutoDiff.logistic(x)
    z.outer()
    assert_array_almost_equal(z.get_val(), np.array([0.88079708, 0.95257413]))
    assert_array_almost_equal(x.get_grad(), np.array([0.10499359, 0.04517666]))

#Test mul_by_row()
def test_mul_by_row():
    assert AutoDiff.mul_by_row(2,1) == 2

# if __name__ == "__main__" :
#     import AutoDiff
#     test_AD_create_f()
#     test_AD_stack_f()
#     test_fAD_constructor_init()
#     test_fAD_add()
#     test_fAD_sub()
#     test_fAD_mul()
#     test_fAD_div()
#     test_fAD_pow()
#     test_fAD_neg()
#     test_fAD_abs()
#     test_fAD_print()
#     test_fAD_len()
#     test_fAD_eq()
#     test_fAD_ne()
#     test_rAD_constructor_init()
#     test_rAD_reset_der()
#     test_rAD_add()
#     test_rAD_sub()
#     test_rAD_mul()
#     test_rAD_div()
#   test_rAD_pow()
#     test_rAD_neg()
#     test_rAD_abs()
#     test_rAD_str()
#     test_combined_sin()
#     test_combined_cos()
#     test_combined_log()
#     test_combined_exp()
#     test_combined_arcsin()
#     test_combined_arccos()
#     test_combined_arctan()
#     test_combined_sinh()
#     test_rAD_create_r()
#     test_combined_exp()
#     test_combined_logistic()
#     test_rAD_eq()
#     test_rAD_ne()
#     test_mul_by_row()





