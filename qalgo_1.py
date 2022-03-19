import scipy
import numpy as np
import itertools

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
# print(list(g1))
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

# checks
for i in funcs:
    if len(i) != 256 or len(i[0]) != 256:
        print("not 256!")

    eig = np.linalg.eigvals(i)
    if list(eig).count(1) + list(eig).count(-1) != 256:
        print("eigenvals other than -1 or 1!")

# evaluating
# M=G1G6G7G8G3G2G9G10G4G11G12G3G5G12
M = [g1, g6, g7, g8, g3, g2, g9, g10, g4, g11, g12, g3, g5, g12]

# zero = [0, 1]
# one = [1, 0]

zero = [[0], [1]]
one = [[1], [0]]


gen = list(itertools.product([zero, one], repeat=3))

for ii in gen:
    inp = tenz(np.array([zero, *ii, zero, zero, zero, zero]))
    for i in M[::-1]:
        inp = np.matmul(i, inp)


    print(inp[5])
print(inp)

