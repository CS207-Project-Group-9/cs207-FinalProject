from AD import *

# a = AutoDiff('a',[1,0])
# >> TypeError('First argument needs to be consisted of numbers')

# a = AutoDiff(2,[1.5,'a'])
# >> TypeError('Second argument needs to be consisted of numbers')

# a = AutoDiff([1,3],[[0,1],[1]])
# >> ValueError('Input dimensions do not match')

# a = AutoDiff([1,3],[[1,0,0],[0,1,0],[0,0,1]])
# >> ValueError('Input dimensions do not match')


a = AutoDiff(1,[1,2,2])
x,y,z = AD_create([1,2,3])

# print(z.val)
# >> [3]

new_AD = AD_stack([x,y,z])

# print('{},{}'.format(new_AD.val, new_AD.der))
# >> [1 2 3],[[1 0 0],[0 1 0],[0 0 1]]

new_new_AD = AD_stack([new_AD, a])

# print('{},{}'.format(new_new_AD.val, new_new_AD.der))
# >> [1 2 3 1],[[1 0 0],[0 1 0],[0 0 1],[1 2 2]]

test1 = x + y
#print('{},{}'.format(test1.val, test1.der))
# >> [3],[[1 1 0]]

test2 = x - y
#print('{},{}'.format(test2.val, test2.der))
# >> [-1],[[1 -1 0]]

test3 = x * y
# print('{},{}'.format(test3.val, test3.der))
# >> [2],[[2 1 0]]

test4 = x / y
# print('{},{}'.format(test4.val, test4.der))
# >> [0.5],[[ 0.5  -0.25  0.  ]]

test5 = 18 / z
# print('{},{}'.format(test5.val, test5.der))
# >> [6.],[[ 0.  0.  -2.]]

test6 = test3 + test4
# print('{},{}'.format(test6.val, test6.der))
# >> [2.5],[[2.5  0.75 0.  ]]

test7 = test2 * test5
# print('{},{}'.format(test7.val, test7.der))
# >> [-6.],[[ 6. -6.  2.]]

test8 = test7 ** 3
# print('{},{}'.format(test8.val, test8.der))
# >> [-216.],[[ 648. -648.  216.]]

test9 = 3 ** test3
# print('{},{}'.format(test9.val, test9.der))
# >> [9],[[19.7750212  9.8875106  0.       ]]

test10 = test2.sin()
# print('{},{}'.format(test10.val, test10.der))
# >> [-0.84147098],[[ 0.54030231 -0.54030231  0.        ]]

test11 = test2.cos()
# print('{},{}'.format(test11.val, test11.der))
# >> [0.54030231],[[ 0.84147098 -0.84147098  0.        ]]

test12 = -test7
# print('{},{}'.format(test12.val, test12.der))
# >> [6.],[[-6.  6. -2.]]

test13 = test12.log()
# print('{},{}'.format(test13.val, test13.der))
# >> [1.79175947],[[-1.          1.         -0.33333333]]
