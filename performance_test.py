#!/usr/bin/env python3
import os
import sys
from random import *
import numpy as np

MAX = 1e3
SCALE = 1.1
p = 4
try:
    p = int(sys.argv[1])
except: pass
n = 10
# define matrix A and vector b

A = np.random.uniform(0, MAX, (n, n))
x = np.random.uniform(0, MAX, n)
print(x)

# b = np.dot(A, x)

d = sum(A[i][i] for i in range(len(A)))
o = np.sum(A) - d
t = o / d * SCALE
for i in range(len(A)):
     A[i][i] *= t

N = np.random.choice((-1, 1), A.shape)
A *= N
b = A@x

in_fname1 = f'input_mat_{n}.txt'
in_fname2 = f'input_vec_{n}.txt'
with open(in_fname1, 'w') as f:
        f.write(f'{n}\n')
        for i in range(n):
            f.write(' '.join(map(str, A[i, :])) + '\n')

with open(in_fname2, 'w') as f:
    f.write(' '.join(map(str, b)) + '\n')

out_fname = f'output_{p}.txt'


os.system(f'make && mpirun -np {p} ./pjacobi {in_fname1} {in_fname2} {out_fname}> /dev/null')

with open(out_fname, 'r') as f:
    prog_out = list(map(float, f.readline().strip().split()))

# x_actual = np.linalg.solve(A, b)
# compare prog out with actual x
if np.allclose(np.array(prog_out), x, rtol=1e-9, atol=1e-9):
    print('The program output is correct!')
else:
    print('The program output is incorrect.')

# os.remove(in_fname1)
# os.remove(in_fname2)
# os.remove(out_fname)

