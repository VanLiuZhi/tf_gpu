import numpy as np

# a = [[2, 9], [3, 6], [4, 7]]
#
# res = np.array([1, 2, 3])
#
# a = np.argmax(a, axis=1)
# print(a)
# print(res.shape)

# import numpy as np

a = [[2, 9], [3, 6]]
print(a)

res = np.argmax(a, axis=1, out=np.array([3, 2]))

print(res)
