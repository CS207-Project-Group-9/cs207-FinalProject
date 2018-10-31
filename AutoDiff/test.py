from AD import *
a = AutoDiff(1,[1,2,2])
x,y,z = AD_create([1,2,3])
print(z.val)
new_AD = AD_stack([x,y,z])
print('{},{}'.format(new_AD.val, new_AD.der))
new_new_AD = AD_stack([new_AD, a])
print('{},{}'.format(new_new_AD.val, new_new_AD.der))