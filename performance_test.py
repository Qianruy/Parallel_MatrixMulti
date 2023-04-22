#!/usr/bin/env python3
import os
import sys
from random import *
import numpy as np

MAX = 1e3
SCALE = 1.2

q = 2
try:
    q = int(sys.argv[1])
except: pass

n = 10
# define matrix A and vector b
A = np.random.uniform(0, MAX, (n, n))
x = np.random.uniform(0, MAX, n)

for i in range(len(A)):
    diagnal = A[i][i] 
    other = sum(A[i]) - diagnal
    A[i][i] *= other / diagnal * SCALE

# randomly multiply 1 or (-1)
A *= np.random.choice((-1, 1), A.shape)
b = np.dot(A, x)

in_fname1 = f'input_mat_{n}.txt'
in_fname2 = f'input_vec_{n}.txt'
with open(in_fname1, 'w') as f:
        f.write(f'{n}\n')
        for i in range(n):
            f.write(' '.join(map(str, A[i, :])) + '\n')

with open(in_fname2, 'w') as f:
    f.write(' '.join(map(str, b)) + '\n')

out_fname = f'output.txt'

os.system(f'make && mpirun -np {q*q} ./pjacobi {in_fname1} {in_fname2} {out_fname}> /dev/null')

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

