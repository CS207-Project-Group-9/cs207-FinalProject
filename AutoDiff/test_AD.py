#test_AD.py
#Nov 6, 2018

#This test suite is associated with file 'AD.py', 
#which implements forward-mode automatic differentiation.

#import unit testing packages pytest and numpy testing
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
#import AD
from AutoDiff import AD

#AD_create allows for simultaneous assignment 
#of AD instances
def test_AD_create():
    a = AD.create(3)
    assert a.val == [3], a.der == [[1]]
    a, b, c = AD.create([1, 2, 3])
    assert a.val == [1], a.der == [[1,0,0]]
    assert b.val == [2], b.der == [[0,1,0]]
    assert c.val == [3], c.der == [[0,0,1]]
    a, b = AD.create([[1, 2],[3, 4]])
    assert a.val == [1, 2], a.der == [[1, 0], [1, 0]]
    assert b.val == [3, 4], b.der == [[0, 1], [0, 1]]



#AD_stack takes in multiple AD instances
#in the form of numpy arrays, returns 
#values as a vector and derivatives as a matrix
def test_AD_stack():
    a, b, c = AD.create([1, 2, 3])
    c = AD.stack([a, b, c])
    assert_array_equal(c.val, np.array([1,2,3]))
    assert_array_equal(c.der, np.array([[1,0,0],[0,1,0],[0,0,1]]))

#Test whether constructor of AutoDiff class 
#returns proper values, derivatives, and errors
def test_AutoDiff_constuctor_init():
    a = 5.0
    b = AD.AutoDiff(a,1)
    assert_array_equal(b.val, np.array([5.0]))
    assert_array_equal(b.der, np.array([[1]]))
    assert_array_equal(b.get_val(), np.array([5.0])) #test get_val()
    assert_array_equal(b.get_der(), np.array([[1]])) #test get_der()
    #inputs ought not to be type other than integer, list or numpy array
    with pytest.raises(TypeError):
        AD.AutoDiff('hello','friend')
    with pytest.raises(TypeError):
        AD.AutoDiff([5.0], 'test')
    #check maximal dimensions of val and der
    with pytest.raises(ValueError):
        AD.AutoDiff([[1],[2]], [[1,0],[0,1]])
    with pytest.raises(ValueError):
        AD.AutoDiff([[5.0]], [[[1,0,0],[0,1,0]]])
    #check if dimension of derivative input matches that of value input
    with pytest.raises(ValueError):
        AD.AutoDiff([1], [[1],[2]])
    with pytest.raises(ValueError):
        AD.AutoDiff([1,2], [1,2,3,4])

    #check variable type
    with pytest.raises(TypeError):
        AD.AutoDiff([1,2], [['a','b','c'],['d','e','f']])
    with pytest.raises(TypeError):
        AD.AutoDiff(['a'], [[1,2]])

    with pytest.raises(ValueError):
        AD.AutoDiff([1,2,3],[[1,2,3],[1,2]])
    with pytest.raises(ValueError):
        AD.AutoDiff([1,2,3],[[1,0,0],[0,1,0]])
    
    

        
#Test whether addition works between AD instances, 
#and between AD instance and number, regardless of order
def test_AutoDiff_add():
    x, y = AD.create([5.0, 7.0])
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
def test_AutoDiff_sub():
    x, y = AD.create([5.0, 7.0])
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
def test_AutoDiff_mul():
    x, y = AD.create([5.0, 7.0])
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
def test_AutoDiff_div():
    x, y = AD.create([4.0, 8.0])
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
def test_AutoDiff_pow():
    x, y = AD.create([2.0, 3.0])
    a, b = AD.create(np.array([1.0, 2.0]))
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
def test_AutoDiff_neg():
    x, y = AD.create([2.0, 8.0])
    neg1 = -x
    neg2 = -(x/y)
    assert neg1.val == [-2.0]
    assert_array_equal(neg1.der, np.array([[-1, 0]]))
    assert neg2.val == [-0.25]
    assert_array_equal(neg2.der, np.array([[-0.125, 0.03125]]))

#Test whether taking the sine of AD instance returns the correct value
#Test whether the sin() function also apply to integers
def test_AutoDiff_sin():
    x = AD.AutoDiff(1.0, [1, 0])
    y = AD.sin(x)
    assert_array_almost_equal(y.val, np.array([0.84147098]), decimal = 6)
    assert_array_almost_equal(y.der, np.array([[0.54030231, 0.]]), decimal = 6)
    a = 6.0
    b = AD.sin(a)
    assert b == -0.27941549819892586

#Test whether taking the cosine of AD instance returns the correct value
def test_AutoDiff_cos():
    a, b = AD.create([2.0, 8.0])
    c = AD.cos(a*b)
    assert_array_almost_equal(c.val, np.array([-0.95765948]), decimal = 6)
    assert_array_almost_equal(c.der, np.array([[2.30322653, 0.57580663]]), decimal = 6)
    x = 5.0
    y = AD.cos(x)
    assert y == pytest.approx(0.2836621854632263)

#Test whether taking the natural logarithm of AD instance returns the correct value
def test_AutoDiff_log():
    a, b = AD.create([-4.0, 8.0])
    assert_array_almost_equal(AD.log(b).val, np.array([2.07944154]), decimal = 6)
    assert_array_equal(AD.log(b).der, np.array([[0, 0.125]]))
    with pytest.raises(ValueError):
        AD.log(a)
    x = 5.0
    y = -4.0
    assert AD.log(x) == 1.6094379124341003
    with pytest.raises(ValueError):
        AD.log(y)

#Test exp()
def test_AutoDiff_exp():
    x, y = AD.create([2.0, 3.0])
    z = AD.exp(x)
    a = 5.0
    b = AD.exp(a)
    assert_array_almost_equal(z.val, np.array([7.3890561]))
    assert_array_almost_equal(z.der, np.array([[7.3890561, 0. ]]))
    assert b == 148.4131591025766

#Test __abs__
def test_AutoDiff_abs():
    a = AD.AutoDiff(-8)
    b = abs(a)
    assert b.val == [8]
    assert b.der == [[-1.]]
        
#Test __str__ and __repr__
def test_AutoDiff_print():
    a, b = AD.create([2.0, 8.0])
    assert 'AutoDiff Object' in str(a)
    assert 'AutoDiff' in repr(b)
#     assert str(a) == 'AutoDiff Object, val: [2.], der: [[1 0]]'
#     assert repr(b) == 'AutoDiff([8.],[[0 1]])'

#Test __len__
def test_AutoDiff_len():
    a, b = AD.create([2.0, 8.0])
    c = AD.stack([a,b])
    assert len(c) == 2

#Test __eq__
def test_AutoDiff_eq():
    a = AD.AutoDiff(8.0)
    b = AD.AutoDiff(8.0)
    c = AD.AutoDiff(5.0)
    assert a == b
    assert (a == c) == False