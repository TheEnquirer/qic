import scipy
import numpy as np

from functools import partial

class Infix(object):
    def __init__(self, func):
        self.func = func
    def __or__(self, other):
        return self.func(other)
    def __ror__(self, other):
        return Infix(partial(self.func, other))
    def __call__(self, v1, v2):
        return self.func(v1, v2)
# @Infix
# def t(x, y):
#     # return np.kron(x[0], y[0])
#     return np.kron(x, y)

tensor = np.kron
i = np.identity

# blocks
x = np.array([[0, 1],
              [1, 0]])
s = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 1]])
tof = np.array([
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,1,0]])

def tenz(arr):
    cur = arr[0]
    for i in arr[1:]:
        cur = tensor(cur, i)
    return cur



# functions
# G1 = I8 ⊗ TOF ⊗ I4,
g1 = tenz([i(8), tof, i(4)])

# G2 = TOF ⊗ I2 ⊗ TOF ⊗ I2,
g2 = tenz([tof, i(2), tof, i(2)])

# G3 = I4 ⊗ X ⊗ I32,
g3 = tenz([i(4), x, i(32)])

# G4 = I4 ⊗ TOF ⊗ I8,
g4 = tenz([i(4), tof,i(8)])

# G5 = I2 ⊗ TOF ⊗ I16,
g5 = tenz([i(2), tof, i(16)])

# G6 = I32 ⊗ S ⊗ I2,
g6 = tenz([i(32), s, i(2)])

# G7 = I16 ⊗ S ⊗ S,
g7 = tenz([i(16), s, s])

# G8 = I4 ⊗ S ⊗ I2 ⊗ S ⊗ I2,
g8 = tenz([i(4), s, i(2), s, i(2)])

# G9 = I2 ⊗ S ⊗ I32,
g9 = tenz([i(2), s, i(32)])

# G10 = S ⊗ S ⊗ X ⊗ I8,
g10 = tenz([i(2), s, i(32)])

# G11 = I16 ⊗ S ⊗ I4,
g11 = tenz([i(16), s, i(4)])

# G12 = I8 ⊗ S ⊗ I8
g12 = tenz([i(8), s, i(8)])

funcs = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12]

for i in funcs:
    if len(i) != 256 or len(i[0]) != 256:
        print("not 256!")


# g1 = (i(8) |tensor| tof)
# l = np.kron(i(8), tof)
# print(l |tensor| i(4))
# print(g1)
# l = tensor(tof, i(4))
# g1 = tensor(i(8), tensor(tof, i(4)))
# g1 = [i(8)] |t| ([tof] |t| [i(4)])

# G2 = TOF ⊗ I2 ⊗ TOF ⊗ I2,
# g2 = ((
    # ([tof] |t| [i(2)]) |t| [tof]) |t| [i(2)]
# g2 = ([tof] |t| [i(2)]) |t| [tof]


